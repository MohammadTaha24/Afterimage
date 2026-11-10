"""Build the Word manual from README.md (requires python-docx).

Run in a documentation environment, not the model environment. Existing terminal
and app screenshots are retained from the manual unless --screenshots is supplied
with another .docx as the screenshot source. The README is the text source of truth.
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent


def inline(paragraph, text):
    for part in re.split(r"(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))", text):
        if not part:
            continue
        link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part)
        if link:
            hyperlink = OxmlElement("w:hyperlink")
            hyperlink.set(qn("r:id"), paragraph.part.relate_to(
                link[2], "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True))
            run = OxmlElement("w:r")
            props = OxmlElement("w:rPr")
            style = OxmlElement("w:rStyle")
            style.set(qn("w:val"), "Hyperlink")
            props.append(style)
            run.append(props)
            value = OxmlElement("w:t")
            value.text = link[1]
            run.append(value)
            hyperlink.append(run)
            paragraph._p.append(hyperlink)
        elif part.startswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif part.startswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        else:
            paragraph.add_run(part)


def cell_border(cell):
    props = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            el.set(qn(f"w:{key}"), value)
        borders.append(el)
    props.append(borders)
    margins = OxmlElement("w:tcMar")
    for name in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{name}")
        el.set(qn("w:w"), "100")
        el.set(qn("w:type"), "dxa")
        margins.append(el)
    props.append(margins)


def build(source, output, screenshot_source):
    images = {}
    if screenshot_source.exists():
        with zipfile.ZipFile(screenshot_source) as archive:
            for name in ("word/media/image3.png", "word/media/image4.png"):
                if name in archive.namelist():
                    images[name] = archive.read(name)
            if not images:
                # A generated manual renumbers the two retained images to 1 and 2.
                media = [name for name in archive.namelist() if name.startswith("word/media/") and name.endswith(".png")]
                if len(media) == 2:
                    images = {name: archive.read(name) for name in sorted(media)}
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.7)
    section.left_margin = section.right_margin = Inches(0.75)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.06
    for name, size in (("Title", 26), ("Heading 1", 16), ("Heading 2", 12)):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(13 if name != "Title" else 0)
        style.paragraph_format.space_after = Pt(7)
        for border in style.element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    doc.styles["Caption"].font.color.rgb = RGBColor(0, 0, 0)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    pos = 0
    while pos < len(lines):
        line = lines[pos].strip()
        pos += 1
        if not line:
            continue
        if line.startswith("```"):
            commands = []
            while pos < len(lines) and not lines[pos].strip().startswith("```"):
                commands.append(lines[pos])
                pos += 1
            pos += 1
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(9)
            p.paragraph_format.keep_together = True
            p.paragraph_format.line_spacing = 1.05
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "F3F4F5")
            p._p.get_or_add_pPr().append(shade)
            run = p.add_run("\n".join(commands))
            run.font.name, run.font.size = "Consolas", Pt(9)
            continue
        if line.startswith("|"):
            rows = [line]
            while pos < len(lines) and lines[pos].strip().startswith("|"):
                rows.append(lines[pos].strip())
                pos += 1
            rows = [row for row in rows if not re.fullmatch(r"[| :\-]+", row)]
            table = doc.add_table(rows=0, cols=2)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = False
            for index, row in enumerate(rows):
                cells = table.add_row().cells
                cells[0].width, cells[1].width = Inches(2), Inches(5)
                for cell, text in zip(cells, row.strip("|").split("|")):
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    cell_border(cell)
                    inline(cell.paragraphs[0], text.strip())
                    cell.paragraphs[0].paragraph_format.space_after = Pt(3)
                    if index == 0:
                        shade = OxmlElement("w:shd")
                        shade.set(qn("w:fill"), "E7ECF0")
                        cell._tc.get_or_add_tcPr().append(shade)
                        for run in cell.paragraphs[0].runs:
                            run.bold = True
                if index == 0:
                    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
            doc.add_paragraph().paragraph_format.space_after = Pt(0)
            continue
        if line.startswith("# "):
            doc.add_paragraph(line[2:], "Title")
        elif line.startswith("## "):
            # Section numbers are ordinary text; keep the heading labels readable.
            doc.add_paragraph(line[3:].replace(".", "", 1) if re.match(r"\d+\.", line[3:]) else line[3:], "Heading 1")
        elif line.startswith("### "):
            doc.add_paragraph(line[4:], "Heading 2")
        elif line.startswith("- "):
            inline(doc.add_paragraph(style="List Bullet"), line[2:])
        elif re.match(r"\d+\. ", line):
            # Preserve the explicit numbering from Markdown to avoid numbering restarts.
            inline(doc.add_paragraph(), line)
        else:
            inline(doc.add_paragraph(), line)
    if images:
        doc.add_page_break()
        doc.add_paragraph("Application reference", "Heading 1")
        captions = ("The backend and frontend use separate terminals with the same activated environment.",
                    "Select a model and confirm loading before uploading an image.")
        for (name, data), caption in zip(images.items(), captions):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(io.BytesIO(data), width=Inches(6.5))
            cap = doc.add_paragraph(caption, "Caption")
            cap.paragraph_format.space_after = Pt(14)
    doc.core_properties.title = "Afterimage User Manual"
    doc.core_properties.subject = "Installation and runtime verification"
    doc.core_properties.author = "Afterimage"
    doc.save(output)
    print(f"Built {output} from {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "README.md")
    parser.add_argument("--output", type=Path, default=ROOT / "Afterimage_User_Manual.docx")
    parser.add_argument("--screenshots", type=Path, help="DOCX containing the original screenshots")
    args = parser.parse_args()
    build(args.source, args.output, args.screenshots or args.output)
