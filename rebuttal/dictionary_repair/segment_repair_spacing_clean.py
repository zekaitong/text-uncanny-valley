#!/usr/bin/env python3
"""Clean whitespace-fragmentation repair with external dictionaries only.

This script is a stricter version of ``segment_repair_spacing.py`` for
mitigation experiments. It deliberately avoids any benchmark/domain-specific
hand-written vocabulary. The only lexical resource is an external dictionary
file such as Hunspell's ``en_US.dic`` or ``/usr/share/dict/words``.

The script preserves the input JSONL schema and rewrites only string entries in
the ``lines`` field. It writes new files by default and never overwrites inputs
unless ``--overwrite`` is passed for an existing output path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_INPUTS = [
    *(f"data/github_prs_100_wa{i}.jsonl" for i in range(1, 11)),
    *(f"data/gsm8k_n80_wa{i}.jsonl" for i in range(1, 11)),
    *(f"data/legal_docs_w{i}.jsonl" for i in range(1, 11)),
]

DEFAULT_DICTIONARY_CANDIDATES = [
    "/usr/share/hunspell/en_US.dic",
    "/usr/share/dict/words",
]

TOKEN_RE = re.compile(r"[A-Za-z]+|[ \t]+|[^A-Za-z \t]+")


def strip_hunspell_flags(word: str) -> str:
    return word.split("/", 1)[0].strip()


def load_words(paths: Sequence[str]) -> tuple[set[str], list[str]]:
    words: set[str] = set()
    sources: list[str] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        loaded = 0
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line_no, line in enumerate(handle, 1):
                word = strip_hunspell_flags(line)
                if line_no == 1 and word.isdigit():
                    continue
                if not word or not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", word):
                    continue
                word = word.replace("'", "").replace("-", "").lower()
                if 1 <= len(word) <= 40:
                    words.add(word)
                    loaded += 1
        if loaded:
            sources.append(str(path))

    words.update({"a", "i"})
    if not sources:
        raise FileNotFoundError(
            "No external dictionary found. Pass --dictionary /path/to/wordlist. "
            "This clean repair script intentionally has no fallback/domain vocabulary."
        )
    return words, sources


@dataclass
class RepairConfig:
    words: frozenset[str]
    max_join_tokens: int = 8
    max_join_chars: int = 48
    max_word_len: int = 28
    repair_numbers: bool = True
    segment_cache: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class RepairStats:
    rows: int = 0
    lines: int = 0
    changed_lines: int = 0

    def add_line(self, changed: bool) -> None:
        self.lines += 1
        if changed:
            self.changed_lines += 1


def is_known(word: str, words: frozenset[str]) -> bool:
    lower = word.lower()
    if lower in words:
        return True
    if len(lower) > 3 and lower.endswith("s") and lower[:-1] in words:
        return True
    if len(lower) > 4 and lower.endswith("ed") and lower[:-2] in words:
        return True
    if len(lower) > 5 and lower.endswith("ing") and lower[:-3] in words:
        return True
    return False


def has_fragment_evidence(tokens: Sequence[str], words: frozenset[str]) -> bool:
    if len(tokens) < 2:
        return False
    for token in tokens:
        lower = token.lower()
        if lower not in {"a", "i"} and len(lower) <= 2:
            return True
        if not is_known(lower, words):
            return True
    return False


def segment_word_cost(word: str, cfg: RepairConfig) -> float | None:
    if word in {"a", "i"}:
        return 0.2
    if not is_known(word, cfg.words):
        return None
    if len(word) <= 2:
        return 5.0
    if len(word) == 3:
        return 2.3
    return 1.2 + 0.02 * len(word)


def greedy_segment_lower(text: str, cfg: RepairConfig) -> list[str]:
    lower = text.lower()
    if not lower:
        return []
    cached = cfg.segment_cache.get(lower)
    if cached is not None:
        return list(cached)

    direct_cost = segment_word_cost(lower, cfg)
    if direct_cost is not None and len(lower) > 3:
        cfg.segment_cache[lower] = (lower,)
        return [lower]

    n = len(lower)
    inf = 10**9
    dp: list[tuple[float, list[str]]] = [(inf, []) for _ in range(n + 1)]
    dp[n] = (0.0, [])

    for pos in range(n - 1, -1, -1):
        max_end = min(n, pos + cfg.max_word_len)
        for end in range(pos + 1, max_end + 1):
            word = lower[pos:end]
            cost = segment_word_cost(word, cfg)
            if cost is None:
                continue
            next_cost, next_words = dp[end]
            if next_cost >= inf:
                continue
            total = cost + next_cost + (0.03 if end < n else 0.0)
            candidate_words = [word, *next_words]
            best_cost, best_words = dp[pos]
            if total < best_cost - 1e-9 or (
                abs(total - best_cost) <= 1e-9 and len(candidate_words) < len(best_words)
            ):
                dp[pos] = (total, candidate_words)

    pieces = dp[0][1] if dp[0][0] < inf else [lower]
    cfg.segment_cache[lower] = tuple(pieces)
    return pieces


def restore_case(joined_original: str, lower_segments: Sequence[str]) -> list[str]:
    if not lower_segments:
        return []
    if len(lower_segments) == 1:
        return [joined_original]

    restored: list[str] = []
    pos = 0
    for segment in lower_segments:
        part = joined_original[pos : pos + len(segment)]
        restored.append(part if len(part) == len(segment) else segment)
        pos += len(segment)
    return restored


def keep_cost(token: str, cfg: RepairConfig) -> float:
    lower = token.lower()
    if lower in {"a", "i"}:
        return 0.05
    if len(lower) <= 2:
        return 3.0 if is_known(lower, cfg.words) else 4.0
    if is_known(lower, cfg.words):
        return 0.1
    if token[:1].isupper():
        return 1.0 + 0.03 * len(token)
    return 2.0 + 0.08 * len(token)


def repair_candidate(tokens: Sequence[str], cfg: RepairConfig) -> tuple[list[str], float] | None:
    if len(tokens) < 2 or not has_fragment_evidence(tokens, cfg.words):
        return None

    joined = "".join(tokens)
    if len(joined) > cfg.max_join_chars:
        return None

    lower_segments = greedy_segment_lower(joined, cfg)
    if not lower_segments:
        return None
    output = restore_case(joined, lower_segments)

    known_segments = sum(1 for part in lower_segments if is_known(part, cfg.words))
    unknown_chars = sum(len(part) for part in lower_segments if not is_known(part, cfg.words))

    if len(output) == 1 and not is_known(output[0], cfg.words):
        short_fragments = sum(1 for token in tokens if len(token) <= 2)
        if short_fragments == 0:
            return None
        cost = 1.4 + 0.08 * len(joined) + 0.15 * len(tokens)
        return output, cost

    cost = 0.25 * len(output) + 0.15 * (len(tokens) - 1) + 1.2 * unknown_chars
    if known_segments == 0:
        cost += 8.0
    return output, cost


def suspicious_token(token: str, cfg: RepairConfig) -> bool:
    lower = token.lower()
    if lower not in {"a", "i"} and len(lower) <= 2:
        return True
    return not is_known(lower, cfg.words)


def repair_word_sequence(tokens: Sequence[str], cfg: RepairConfig) -> list[str]:
    n = len(tokens)
    if n <= 1:
        return list(tokens)

    repaired: list[str] = []
    i = 0

    while i < n:
        nearby_suspicious = suspicious_token(tokens[i], cfg) or (
            i + 1 < n and suspicious_token(tokens[i + 1], cfg)
        )
        if not nearby_suspicious:
            repaired.append(tokens[i])
            i += 1
            continue

        best_j = i + 1
        best_output: list[str] | None = None
        best_improvement = 0.0

        max_j = min(n, i + cfg.max_join_tokens)
        for j in range(i + 2, max_j + 1):
            group = tokens[i:j]
            if sum(len(token) for token in group) > cfg.max_join_chars:
                break
            candidate = repair_candidate(group, cfg)
            if candidate is None:
                continue
            candidate_output, candidate_cost = candidate
            if [part.lower() for part in candidate_output] == [part.lower() for part in group]:
                continue
            baseline_cost = sum(keep_cost(token, cfg) for token in group)
            improvement = baseline_cost - candidate_cost
            if improvement > best_improvement + 0.05:
                best_improvement = improvement
                best_j = j
                best_output = candidate_output

        if best_output is not None:
            repaired.extend(best_output)
            i = best_j
        else:
            repaired.append(tokens[i])
            i += 1

    return repaired


def process_soft_span(span: list[str], cfg: RepairConfig) -> str:
    leading: list[str] = []
    trailing: list[str] = []
    inner = list(span)

    while inner and inner[0] == " ":
        leading.append(inner.pop(0))
    while inner and inner[-1] == " ":
        trailing.append(inner.pop())

    words = [piece for piece in inner if piece != " "]
    if len(words) <= 1:
        return "".join(span)
    repaired = repair_word_sequence(words, cfg)
    return "".join(leading) + " ".join(repaired) + "".join(reversed(trailing))


def repair_number_spaces(line: str) -> str:
    line = re.sub(r"(?<=\d) (?=\d)", "", line)
    line = re.sub(r"(?<=\d) (?=[.,:/-]\d)", "", line)
    line = re.sub(r"(?<=[.,:/-]) (?=\d)", "", line)
    return line


def repair_line(line: str, cfg: RepairConfig) -> str:
    tokens = TOKEN_RE.findall(line)
    output: list[str] = []
    soft_span: list[str] = []

    def flush_span() -> None:
        nonlocal soft_span
        if soft_span:
            output.append(process_soft_span(soft_span, cfg))
            soft_span = []

    for token in tokens:
        if re.fullmatch(r"[A-Za-z]+", token):
            soft_span.append(token)
        elif token == " ":
            if soft_span:
                soft_span.append(token)
            else:
                output.append(token)
        else:
            flush_span()
            output.append(token)

    flush_span()
    repaired = "".join(output)
    if cfg.repair_numbers:
        repaired = repair_number_spaces(repaired)
    return repaired


def output_path_for(input_path: Path, output_dir: Path | None, suffix: str) -> Path:
    target_dir = output_dir if output_dir is not None else input_path.parent
    return target_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def repair_file(
    input_path: Path,
    output_path: Path,
    cfg: RepairConfig,
    dictionary_sources: Sequence[str],
    overwrite: bool,
    dry_run: bool,
) -> RepairStats:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    stats = RepairStats()
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")

    out_handle = None
    try:
        if not dry_run:
            out_handle = tmp_path.open("w", encoding="utf-8")

        with input_path.open(encoding="utf-8", errors="replace") as in_handle:
            for raw_line in in_handle:
                if not raw_line.strip():
                    continue
                obj = json.loads(raw_line)
                stats.rows += 1

                original_lines = obj.get("lines")
                record_line_count = 0
                record_changed_lines = 0
                if isinstance(original_lines, list):
                    repaired_lines = []
                    for value in original_lines:
                        if isinstance(value, str):
                            repaired = repair_line(value, cfg)
                            changed = repaired != value
                            stats.add_line(changed)
                            record_line_count += 1
                            if changed:
                                record_changed_lines += 1
                            repaired_lines.append(repaired)
                        else:
                            repaired_lines.append(value)
                    obj["lines"] = repaired_lines

                metadata_key = "metadata" if "metadata" in obj else "_metadata"
                if metadata_key not in obj or not isinstance(obj[metadata_key], dict):
                    obj[metadata_key] = {}
                obj[metadata_key].update(
                    {
                        "repair_method": "clean_greedy_external_dictionary_word_segmentation",
                        "repair_policy": "external_dictionary_only_no_domain_hints_no_fallback_vocab",
                        "repair_source_file": str(input_path),
                        "repair_dictionary_sources": list(dictionary_sources),
                        "repair_line_count": record_line_count,
                        "repair_changed_lines": record_changed_lines,
                        "repair_numbers": cfg.repair_numbers,
                        "repair_script": "scripts/segment_repair_spacing_clean.py",
                    }
                )

                if out_handle is not None:
                    out_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if out_handle is not None:
            out_handle.close()
            tmp_path.replace(output_path)
    except Exception:
        if out_handle is not None and not out_handle.closed:
            out_handle.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return stats


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            paths.extend(sorted(Path().glob(pattern)))
        else:
            paths.append(Path(pattern))
    return paths


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean repair of spacing-fragmented JSONL lines using external dictionaries only."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input JSONL files or glob patterns. Defaults to the wa/w1-w10 corrupted files for the three domains.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_clean_wordseg_repair",
        help="Suffix inserted before .jsonl for repaired outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to writing next to each input file.",
    )
    parser.add_argument(
        "--dictionary",
        nargs="+",
        default=DEFAULT_DICTIONARY_CANDIDATES,
        help="External dictionary files, one word per line. Hunspell flags are accepted.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Compute stats without writing outputs.")
    parser.add_argument("--max-join-tokens", type=int, default=8)
    parser.add_argument("--max-join-chars", type=int, default=48)
    parser.add_argument(
        "--no-repair-numbers",
        action="store_true",
        help="Disable conservative numeric whitespace repair.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    words, dictionary_sources = load_words(args.dictionary)
    cfg = RepairConfig(
        words=frozenset(words),
        max_join_tokens=args.max_join_tokens,
        max_join_chars=args.max_join_chars,
        repair_numbers=not args.no_repair_numbers,
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    inputs = expand_inputs(args.inputs)

    if not inputs:
        print("No input files matched.", file=sys.stderr)
        return 2

    print(f"dictionary_words={len(words)} sources={','.join(dictionary_sources)}")
    for input_path in inputs:
        output_path = output_path_for(input_path, output_dir, args.output_suffix)
        stats = repair_file(
            input_path=input_path,
            output_path=output_path,
            cfg=cfg,
            dictionary_sources=dictionary_sources,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        action = "checked" if args.dry_run else "wrote"
        print(
            f"{action}\t{input_path}\t{output_path}\t"
            f"rows={stats.rows}\tlines={stats.lines}\tchanged_lines={stats.changed_lines}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
