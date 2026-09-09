"""
Parse the MedQuAD XML dataset into a single flat CSV:
columns: doc_id, source, focus, url, question, question_type, answer
"""
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import re

ROOT = "/home/claude/MedQuAD"
rows = []

folders = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]

for folder in folders:
    xml_files = glob.glob(os.path.join(ROOT, folder, "*.xml"))
    for path in xml_files:
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            continue
        root = tree.getroot()

        doc_id = root.attrib.get("id", "")
        source = root.attrib.get("source", folder)
        url = root.attrib.get("url", "")
        focus_el = root.find("Focus")
        focus = focus_el.text.strip() if focus_el is not None and focus_el.text else ""

        qa_pairs = root.find("QAPairs")
        if qa_pairs is None:
            continue

        for qa in qa_pairs.findall("QAPair"):
            q_el = qa.find("Question")
            a_el = qa.find("Answer")
            if q_el is None or a_el is None:
                continue
            question = (q_el.text or "").strip()
            qtype = q_el.attrib.get("qtype", "")
            answer = (a_el.text or "").strip()

            if not question or not answer:
                continue

            # Clean up whitespace in answer
            answer_clean = re.sub(r"\s+", " ", answer).strip()
            question_clean = re.sub(r"\s+", " ", question).strip()

            if len(answer_clean) < 20:
                continue

            rows.append({
                "doc_id": doc_id,
                "source": source,
                "focus": focus,
                "url": url,
                "question": question_clean,
                "question_type": qtype,
                "answer": answer_clean,
            })

df = pd.DataFrame(rows)
print(f"Total QA pairs parsed: {len(df)}")
print(df["source"].value_counts())
df.to_csv("/home/claude/medquad_clean.csv", index=False)
print("Saved to /home/claude/medquad_clean.csv")
