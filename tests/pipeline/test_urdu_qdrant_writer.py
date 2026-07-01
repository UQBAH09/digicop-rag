"""
Urdu retrieval validation for bge-m3.

WARNING: Uses synthetic Urdu chunks — not real corpus data.
Must be rerun with real extracted Urdu chunks when available.

Pass criterion (from spec Section 8): hit@1 >= 8/10.
"""

import hashlib
import numpy as np
import pytest

from shared.models import Chunk
from pipeline.embedding.bgeM3_embedder import BgeM3Embedder

# tests/pipeline/urdu_validation_data.py
"""
Synthetic Urdu chunks and questions for bge-m3 retrieval validation.

WARNING: These are synthetic — generated for testing pipeline mechanics only.
This does NOT substitute for validation against real corpus data.
When Urdu extraction is available, rerun this test with real chunks.
"""

URDU_CHUNKS = [
    {
        "id": "ur-001",
        "content": "پانی ایک بے رنگ، بے بو اور بے ذائقہ مائع ہے۔ اس کا کیمیائی فارمولا H2O ہے۔ پانی زندگی کے لیے ضروری ہے اور زمین کی سطح کا تقریباً 71 فیصد حصہ پانی سے ڈھکا ہوا ہے۔",
        "question": "پانی کا کیمیائی فارمولا کیا ہے؟"
    },
    {
        "id": "ur-002",
        "content": "ضیاء ترکیب وہ عمل ہے جس میں پودے سورج کی روشنی، پانی اور کاربن ڈائی آکسائیڈ کو استعمال کرکے خوراک بناتے ہیں۔ اس عمل میں آکسیجن خارج ہوتی ہے۔",
        "question": "ضیاء ترکیب کے عمل میں کون سی گیس خارج ہوتی ہے؟"
    },
    {
        "id": "ur-003",
        "content": "نیوٹن کا پہلا قانون حرکت کہتا ہے کہ کوئی بھی چیز اپنی حالت میں تبدیلی نہیں کرتی جب تک کوئی بیرونی قوت اس پر عمل نہ کرے۔ اسے قانون جمود بھی کہتے ہیں۔",
        "question": "نیوٹن کے پہلے قانون کو اور کیا کہتے ہیں؟"
    },
    {
        "id": "ur-004",
        "content": "مغل سلطنت کی بنیاد 1526 میں بابر نے رکھی۔ پانی پت کی پہلی جنگ میں بابر نے ابراہیم لودھی کو شکست دی اور ہندوستان میں مغل حکومت قائم کی۔",
        "question": "مغل سلطنت کی بنیاد کس نے رکھی؟"
    },
    {
        "id": "ur-005",
        "content": "اردو زبان کا آغاز برصغیر میں ہوا۔ یہ فارسی، عربی اور ہندی زبانوں کے ملاپ سے بنی۔ اردو پاکستان کی قومی زبان ہے۔",
        "question": "پاکستان کی قومی زبان کون سی ہے؟"
    },
    {
        "id": "ur-006",
        "content": "دل ایک عضلاتی عضو ہے جو خون کو پورے جسم میں پمپ کرتا ہے۔ انسانی دل چار خانوں پر مشتمل ہوتا ہے۔ دل ایک منٹ میں تقریباً 72 بار دھڑکتا ہے۔",
        "question": "انسانی دل ایک منٹ میں کتنی بار دھڑکتا ہے؟"
    },
    {
        "id": "ur-007",
        "content": "زمین سورج کے گرد ایک چکر 365 دن اور 6 گھنٹے میں مکمل کرتی ہے۔ اسی لیے ہر چار سال بعد فروری میں ایک دن اضافی ہوتا ہے جسے لیپ سال کہتے ہیں۔",
        "question": "لیپ سال کیوں آتا ہے؟"
    },
    {
        "id": "ur-008",
        "content": "قائداعظم محمد علی جناح پاکستان کے بانی اور پہلے گورنر جنرل تھے۔ انہوں نے مسلمانوں کے لیے ایک الگ وطن کے حصول میں اہم کردار ادا کیا۔",
        "question": "پاکستان کے پہلے گورنر جنرل کون تھے؟"
    },
    {
        "id": "ur-009",
        "content": "بجلی ایلیکٹرونز کے بہاؤ سے پیدا ہوتی ہے۔ بجلی کی دو اقسام ہیں: متبادل کرنٹ اور براہ راست کرنٹ۔ گھروں میں متبادل کرنٹ استعمال ہوتی ہے۔",
        "question": "گھروں میں بجلی کی کون سی قسم استعمال ہوتی ہے؟"
    },
    {
        "id": "ur-010",
        "content": "ہوا میں نائٹروجن کی مقدار تقریباً 78 فیصد اور آکسیجن کی مقدار تقریباً 21 فیصد ہے۔ باقی ایک فیصد میں آرگن اور دیگر گیسیں شامل ہیں۔",
        "question": "ہوا میں آکسیجن کی مقدار کتنی فیصد ہے؟"
    },
]


def make_urdu_chunk(entry: dict) -> Chunk:
    chunk_id = hashlib.sha256(entry["id"].encode()).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        chunk_index=0,
        content=entry["content"],
        element_type="text",
        book_id="urdu-validation-book",
        subject="Urdu",
        grade="9",
        board="Federal",
        lang="ur",
        page_start=1,
        page_end=1,
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_urdu_retrieval_hit_at_1():
    embedder = BgeM3Embedder()

    chunks = [make_urdu_chunk(entry) for entry in URDU_CHUNKS]
    questions = [entry["question"] for entry in URDU_CHUNKS]

    # embed all chunks
    chunk_vectors = embedder.embed(chunks)

    # embed all questions — same encode() call, bge-m3 is symmetric
    question_chunks = [
        Chunk(
            chunk_id=hashlib.sha256(f"q-{i}".encode()).hexdigest(),
            chunk_index=i,
            content=question,
            element_type="text",
            book_id="urdu-validation-questions",
            subject="Urdu",
            grade="9",
            board="Federal",
            lang="ur",
            page_start=1,
            page_end=1,
        )
        for i, question in enumerate(questions)
    ]
    question_vectors = embedder.embed(question_chunks)

    hits = 0
    results_table = []

    for i, (question, q_vec) in enumerate(zip(questions, question_vectors)):
        similarities = [cosine_similarity(q_vec, c_vec) for c_vec in chunk_vectors]
        top_index = similarities.index(max(similarities))
        hit = top_index == i  # correct chunk is at index i
        if hit:
            hits += 1
        results_table.append({
            "question": question,
            "expected_chunk_id": URDU_CHUNKS[i]["id"],
            "top_ranked_chunk_id": URDU_CHUNKS[top_index]["id"],
            "hit": hit,
            "top_similarity": round(max(similarities), 4),
        })

    # print results table for the validation document
    print(f"\n\nUrdu Retrieval Validation Results — hit@1: {hits}/10\n")
    print(f"{'Question':<50} {'Expected':<12} {'Got':<12} {'Hit':<6} {'Score'}")
    print("-" * 95)
    for r in results_table:
        print(f"{r['question']:<50} {r['expected_chunk_id']:<12} {r['top_ranked_chunk_id']:<12} {str(r['hit']):<6} {r['top_similarity']}")

    # pass criterion from spec Section 8: hit@1 >= 8/10
    assert hits >= 8, (
        f"Urdu hit@1 = {hits}/10 — below passing threshold of 8/10. "
        f"Report to supervisor before continuing."
    )