#!/usr/bin/env python3
"""
wiki_extract.py — Extract Wikipedia dump to JSONL, parallel.
Replaces broken WikiExtractor.

Usage:
    python wiki_extract.py ~/enwiki-*-pages-articles.xml.bz2 -o ~/wiki-extracted -w 16
"""

import argparse
import bz2
import json
import re
import xml.etree.ElementTree as ET
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path


def clean_wikitext(text):
    """Strip wikitext markup to plain text."""
    # Remove templates {{...}} (nested)
    depth = 0
    result = []
    i = 0
    while i < len(text):
        if text[i:i+2] == '{{':
            depth += 1
            i += 2
        elif text[i:i+2] == '}}':
            depth = max(0, depth - 1)
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    text = ''.join(result)

    # Remove files/images/categories
    text = re.sub(r'\[\[(?:File|Image|Category):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    # [[link|display]] -> display, [[link]] -> link
    text = re.sub(r'\[\[[^|\]]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # External links [url text] -> text
    text = re.sub(r'\[https?://\S+\s+([^\]]*)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # Refs
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    # HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Bold/italic
    text = re.sub(r"'{2,}", '', text)
    # Headings
    text = re.sub(r'={2,}(.+?)={2,}', r'\1', text)
    # Bullets and numbered lists
    text = re.sub(r'^[*#]+\s?', '', text, flags=re.MULTILINE)
    # Wiki tables
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def page_generator(dump_path):
    """Stream article pages from bz2 XML dump."""
    open_func = bz2.open if str(dump_path).endswith('.bz2') else open
    with open_func(dump_path, 'rb') as f:
        ns = ''
        for event, elem in ET.iterparse(f, events=('end',)):
            tag = elem.tag
            if '}' in tag:
                if not ns:
                    ns = tag.split('}')[0] + '}'
                tag = tag.split('}')[1]

            if tag == 'page':
                title_el = elem.find(f'{ns}title')
                text_el = elem.find(f'.//{ns}text')
                redirect_el = elem.find(f'{ns}redirect')
                ns_el = elem.find(f'{ns}ns')

                title = title_el.text if title_el is not None else ''
                text = text_el.text if text_el is not None else ''
                page_ns = ns_el.text if ns_el is not None else '0'

                elem.clear()

                if redirect_el is not None:
                    continue
                if page_ns != '0':
                    continue
                if not title or not text:
                    continue

                yield title, text


def worker(queue, output_dir, worker_id, min_text_length):
    """Process pages from queue, write JSONL output files."""
    file_idx = 0
    articles = []
    bytes_written = 0
    max_bytes = 1_000_000

    subdir = output_dir / f"worker_{worker_id:02d}"
    subdir.mkdir(parents=True, exist_ok=True)

    while True:
        item = queue.get()
        if item is None:
            break

        title, wikitext = item
        text = clean_wikitext(wikitext)

        if len(text) < min_text_length:
            continue

        article_json = json.dumps({'title': title, 'text': text}, ensure_ascii=False)
        articles.append(article_json)
        bytes_written += len(article_json.encode('utf-8'))

        if bytes_written >= max_bytes:
            outfile = subdir / f"articles_{file_idx:05d}.jsonl"
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write('\n'.join(articles) + '\n')
            articles = []
            bytes_written = 0
            file_idx += 1

    if articles:
        outfile = subdir / f"articles_{file_idx:05d}.jsonl"
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write('\n'.join(articles) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Extract Wikipedia XML dump to JSONL')
    parser.add_argument('dump', help='Path to enwiki-*-pages-articles.xml.bz2')
    parser.add_argument('-o', '--output', default='./wiki-extracted',
                        help='Output directory (default: ./wiki-extracted)')
    parser.add_argument('-w', '--workers', type=int, default=cpu_count(),
                        help=f'Worker processes (default: {cpu_count()})')
    parser.add_argument('--min-text-length', type=int, default=50,
                        help='Min article text length in chars (default: 50)')
    args = parser.parse_args()

    dump_path = Path(args.dump)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {dump_path}")
    print(f"Output:     {output_dir}")
    print(f"Workers:    {args.workers}")
    print()

    queue = Queue(maxsize=args.workers * 100)

    workers = []
    for i in range(args.workers):
        p = Process(target=worker, args=(queue, output_dir, i, args.min_text_length))
        p.start()
        workers.append(p)

    count = 0
    for title, text in page_generator(dump_path):
        queue.put((title, text))
        count += 1
        if count % 10000 == 0:
            print(f"\r  {count:,} articles queued...", end='', flush=True)

    print(f"\r  {count:,} articles total. Finishing workers...")

    for _ in workers:
        queue.put(None)
    for p in workers:
        p.join()

    print(f"Done. {count:,} articles -> {output_dir}")


if __name__ == '__main__':
    main()
