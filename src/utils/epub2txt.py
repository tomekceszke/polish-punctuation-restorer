#!/usr/bin/env python3
# Converts an epub file to plain txt using only Python stdlib.
# Usage: python3 epub2txt.py <input.epub> <output.txt>

import sys
import zipfile
import posixpath
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser


OPF_NS = 'http://www.idpf.org/2007/opf'


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chunks = []
        self._skip_depth = 0
        self._skip_tags = {'script', 'style'}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.chunks.append(data)

    def get_text(self):
        return ''.join(self.chunks)


def find_opf_path(zf):
    container = zf.read('META-INF/container.xml').decode('utf-8')
    root = ET.fromstring(container)
    ns = 'urn:oasis:names:tc:opendocument:xmlns:container'
    rootfile = root.find(f'.//{{{ns}}}rootfile')
    return rootfile.get('full-path')


def parse_opf(zf, opf_path):
    opf_dir = posixpath.dirname(opf_path)
    content = zf.read(opf_path).decode('utf-8')
    root = ET.fromstring(content)

    manifest = {}
    nav_id = None
    for item in root.findall(f'{{{OPF_NS}}}manifest/{{{OPF_NS}}}item'):
        item_id = item.get('id')
        href = item.get('href')
        media_type = item.get('media-type', '')
        props = item.get('properties', '')
        full_path = posixpath.join(opf_dir, href) if opf_dir else href
        manifest[item_id] = full_path
        if 'nav' in props:
            nav_id = item_id

    spine = []
    for itemref in root.findall(f'{{{OPF_NS}}}spine/{{{OPF_NS}}}itemref'):
        idref = itemref.get('idref')
        if idref != nav_id and idref in manifest:
            spine.append(manifest[idref])

    return spine


def extract_text_from_xhtml(content):
    parser = TextExtractor()
    parser.feed(content.decode('utf-8'))
    text = parser.get_text()
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
    non_empty = [l for l in lines if l]
    return '\n'.join(non_empty)


def epub_to_txt(epub_path, txt_path):
    with zipfile.ZipFile(epub_path) as zf:
        opf_path = find_opf_path(zf)
        spine = parse_opf(zf, opf_path)

        parts = []
        for path in spine:
            try:
                content = zf.read(path)
                parts.append(extract_text_from_xhtml(content))
            except KeyError:
                print(f'  warning: missing {path}', file=sys.stderr)

    full_text = '\n\n'.join(p for p in parts if p)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f'Written: {txt_path}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: epub2txt.py <input.epub> <output.txt>')
        sys.exit(1)
    epub_to_txt(sys.argv[1], sys.argv[2])
