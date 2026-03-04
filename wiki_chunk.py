#!/usr/bin/env python3
"""
wiki_chunk.py — Section-aware chunking for extracted Wikipedia JSONL.

Chunks respect document structure:
  1. Split on section headings (== Heading ==) first
  2. Within sections, split on paragraph boundaries (\n\n)
  3. Only split mid-paragraph as last resort for very long paragraphs
  4. Overlap pulls from end of previous chunk for continuity

Each chunk carries section context in metadata so the embedding
captures "Geography of France" not just a decontextualized paragraph.

Output JSONL fields: chunk_id, title, section, text, word_offset

Usage:
    python wiki_chunk.py ~/wiki-extracted -o ~/wiki-chunked -w 16
"""

import argparse
import json
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path


CHUNK_WORDS = 400
OVERLAP_WORDS = 50
MIN_CHUNK_WORDS = 30

# Match wiki-style section headings that survived our extraction cleanup.
# WikiExtractor strips markup but heading text often survives as bare lines
# that were "== Heading ==" -> "Heading" after our regex in wiki_extract.py.
# We also catch any that leaked through with = signs intact.
HEADING_RE = re.compile(r'^(?:={1,5}\s*(.+?)\s*={1,5}|([A-Z][A-Za-z\s,\-&]+))\s*$')


def split_sections(text):
    """Split article into (section_name, section_text) pairs.
    
    Tries heading markers first. If none found (common after aggressive
    markup stripping), falls back to paragraph-based splitting with
    heuristic heading detection: short standalone lines followed by
    longer text are likely headings.
    """
    # First try: look for lines that are plausible headings.
    # After our extractor, headings appear as short standalone lines
    # between double newlines.
    lines = text.split('\n')
    sections = []
    current_heading = "Introduction"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        
        # Detect heading: short line (≤8 words), not a sentence fragment,
        # Title Case or ALL CAPS, preceded/followed by blank lines
        is_heading = False
        
        # Explicit wiki heading markers that survived
        if re.match(r'^={1,5}\s*.+\s*={1,5}$', stripped):
            heading_text = re.sub(r'^=+\s*|\s*=+$', '', stripped)
            is_heading = True
        # Heuristic: short title-case line that doesn't end in period
        elif (stripped
              and len(stripped.split()) <= 8
              and not stripped.endswith('.')
              and not stripped.endswith(',')
              and stripped[0].isupper()
              and len(stripped) > 2
              and len(stripped) < 80):
            # Only treat as heading if surrounded by content
            heading_text = stripped
            is_heading = True

        if is_heading and current_lines:
            section_text = '\n'.join(current_lines).strip()
            if section_text:
                sections.append((current_heading, section_text))
            current_heading = heading_text
            current_lines = []
        elif is_heading and not current_lines:
            current_heading = heading_text
        else:
            current_lines.append(line)

    # Flush final section
    section_text = '\n'.join(current_lines).strip()
    if section_text:
        sections.append((current_heading, section_text))

    # If we only got one section, the heading detection didn't fire much.
    # That's fine — single-section articles just chunk within "Introduction".
    if not sections:
        sections = [("Introduction", text)]

    return sections


def split_paragraphs(text):
    """Split text on double newlines into paragraph blocks."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def word_len(text):
    return len(text.split())


def make_chunks_for_section(title, section, text):
    """Chunk a single section, respecting paragraph boundaries."""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks = []
    chunk_idx = 0
    current_words = []
    current_texts = []

    def flush(overflow_text=None):
        nonlocal chunk_idx, current_words, current_texts
        if not current_texts:
            return

        chunk_text = '\n\n'.join(current_texts)
        if word_len(chunk_text) < MIN_CHUNK_WORDS and chunks:
            # Append tiny remainder to previous chunk
            prev = chunks[-1]
            chunks[-1] = {
                'chunk_id': prev['chunk_id'],
                'title': title,
                'section': section,
                'text': prev['text'] + '\n\n' + chunk_text,
            }
        elif word_len(chunk_text) >= MIN_CHUNK_WORDS:
            # Prepend overlap from previous chunk
            if chunks and OVERLAP_WORDS > 0:
                prev_words = chunks[-1]['text'].split()
                overlap = ' '.join(prev_words[-OVERLAP_WORDS:])
                chunk_text = overlap + ' [...] ' + chunk_text

            chunks.append({
                'chunk_id': f"{title}::{section}::{chunk_idx}",
                'title': title,
                'section': section,
                'text': chunk_text,
            })
            chunk_idx += 1

        current_words = []
        current_texts = []

    for para in paragraphs:
        para_wc = word_len(para)

        # If single paragraph exceeds chunk size, split it on sentences
        if para_wc > CHUNK_WORDS:
            flush()
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_wc = word_len(sent)
                # If a single "sentence" still exceeds chunk size,
                # hard-split on word boundaries as last resort
                if sent_wc > CHUNK_WORDS:
                    words = sent.split()
                    for i in range(0, len(words), CHUNK_WORDS):
                        segment = ' '.join(words[i:i + CHUNK_WORDS])
                        if len(current_words) + word_len(segment) > CHUNK_WORDS and current_texts:
                            flush()
                        current_words.extend(segment.split())
                        current_texts.append(segment)
                else:
                    if len(current_words) + sent_wc > CHUNK_WORDS and current_texts:
                        flush()
                    current_words.extend(sent.split())
                    current_texts.append(sent)
            flush()
            continue

        # Would adding this paragraph exceed the chunk size?
        if len(current_words) + para_wc > CHUNK_WORDS and current_texts:
            flush()

        current_words.extend(para.split())
        current_texts.append(para)

    flush()
    return chunks


def make_chunks(title, text):
    """Split article into section-aware chunks."""
    sections = split_sections(text)
    all_chunks = []

    for section_name, section_text in sections:
        chunks = make_chunks_for_section(title, section_name, section_text)
        all_chunks.extend(chunks)

    return all_chunks


def process_file(args):
    """Process a single JSONL file."""
    input_path, output_dir = args
    chunks_out = []
    count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = article.get('title', '')
            text = article.get('text', '')
            if not text:
                continue

            chunks = make_chunks(title, text)
            for chunk in chunks:
                chunks_out.append(json.dumps(chunk, ensure_ascii=False))
                count += 1

    if not chunks_out:
        return (None, 0)

    parent = input_path.parent.name
    out_subdir = output_dir / parent
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / input_path.name

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chunks_out) + '\n')

    return (str(out_path), count)


def gather_jsonl_files(input_dir):
    return sorted(Path(input_dir).rglob('*.jsonl'))


def main():
    parser = argparse.ArgumentParser(description='Section-aware Wikipedia chunking')
    parser.add_argument('input', help='Input directory with extracted JSONL')
    parser.add_argument('-o', '--output', default='./wiki-chunked',
                        help='Output directory (default: ./wiki-chunked)')
    parser.add_argument('-w', '--workers', type=int, default=cpu_count(),
                        help=f'Worker processes (default: {cpu_count()})')
    parser.add_argument('--chunk-words', type=int, default=CHUNK_WORDS,
                        help=f'Target words per chunk (default: {CHUNK_WORDS})')
    parser.add_argument('--overlap-words', type=int, default=OVERLAP_WORDS,
                        help=f'Overlap words (default: {OVERLAP_WORDS})')
    args = parser.parse_args()

    import wiki_chunk
    wiki_chunk.CHUNK_WORDS = args.chunk_words
    wiki_chunk.OVERLAP_WORDS = args.overlap_words

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = gather_jsonl_files(input_dir)
    if not files:
        print(f"No .jsonl files found in {input_dir}")
        return

    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Files:      {len(files)}")
    print(f"Workers:    {args.workers}")
    print(f"Chunk size: ~{CHUNK_WORDS} words (~{int(CHUNK_WORDS * 1.3)} tokens)")
    print(f"Overlap:    {OVERLAP_WORDS} words")
    print(f"Strategy:   section-aware > paragraph > sentence")
    print()

    work = [(f, output_dir) for f in files]
    total_chunks = 0
    files_done = 0

    with Pool(args.workers) as pool:
        for out_path, count in pool.imap_unordered(process_file, work):
            total_chunks += count
            files_done += 1
            if files_done % 500 == 0:
                print(f"\r  {files_done:,}/{len(files):,} files, {total_chunks:,} chunks", end='', flush=True)

    print(f"\r  {files_done:,}/{len(files):,} files, {total_chunks:,} chunks total")
    print(f"Done. Output in {output_dir}")

    est_vectors = total_chunks
    est_raw_gb = (est_vectors * 768 * 4) / (1024**3)
    est_pq_gb = est_raw_gb / 64
    print(f"\nEstimated vectors:     {est_vectors:,}")
    print(f"Raw FP32 index size:   {est_raw_gb:.1f} GB")
    print(f"IVF_PQ index size:     {est_pq_gb:.2f} GB")


if __name__ == '__main__':
    main()
