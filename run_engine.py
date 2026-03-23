"""
run_engine.py — سكربت التشغيل الآلي لمحرك مهووس v9.0
======================================================
يعمل في بيئة GitHub Actions بشكل تلقائي كامل.
- يقرأ ملفات المتجر من: input/store/
- يقرأ ملفات المنافسين من: input/competitors/
- يقرأ ملف الماركات من: input/brands/
- يحفظ النتائج في: output/
"""

from __future__ import annotations
import os

# استقرار PyTorch/OpenMP على Windows (قبل تحميل sentence-transformers/faiss)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MAHWOUS_ENCODE_BATCH", "64")
os.environ.setdefault("MAHWOUS_ST_BATCH", "8")

import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── إعداد السجل ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("mahwous-runner")

# ── المسارات ─────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
INPUT_STORE     = BASE_DIR / "input" / "store"
INPUT_COMP      = BASE_DIR / "input" / "competitors"
INPUT_BRANDS    = BASE_DIR / "input" / "brands"
OUTPUT_DIR      = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── متغيرات البيئة ───────────────────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
USE_LLM    = os.environ.get("USE_LLM", "true").lower() == "true"

# ── استيراد المحرك ───────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from logic import (
    FeatureParser, GeminiOracle, MahwousEngine,
    MatchResult, SemanticIndex,
    export_brands_csv, export_salla_csv,
    load_brands, load_competitor_products, load_store_products,
)


@dataclass
class EngineRunResult:
    success: bool
    error: str = ""
    output_paths: list[Path] = field(default_factory=list)
    summary_text: str = ""
    elapsed_sec: float = 0.0
    stats: dict = field(default_factory=dict)


def _load_csv_files(folder: Path) -> list:
    """تحميل كل ملفات CSV من مجلد معين."""
    files = list(folder.glob("*.csv"))
    if not files:
        log.warning(f"⚠️ لا توجد ملفات CSV في: {folder}")
        return []
    log.info(f"📂 وجدت {len(files)} ملف في {folder.name}/")
    return files


def _progress_cb(i: int, total: int, name: str) -> None:
    """شريط التقدم في السجل."""
    if i % 100 == 0 or i < 5:
        pct = i / max(total, 1) * 100
        log.info(f"  ⚙️  [{pct:5.1f}%] {i}/{total} — {name[:50]}")


def _log_cb(msg: str) -> None:
    """تسجيل رسائل المحرك."""
    log.info(msg)


def run_engine_paths(
    store_dir: Path,
    comp_dir: Path,
    brands_dir: Path,
    output_dir: Path,
    *,
    use_llm: Optional[bool] = None,
    gemini_key: Optional[str] = None,
    openai_key: Optional[str] = None,
) -> EngineRunResult:
    """
    تشغيل المحرك على مجلدات الإدخال/الإخراج المحددة.
    يُستخدم من سطر الأوامر أو من تطبيق الواجهة (Streamlit).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_llm is None:
        use_llm = os.environ.get("USE_LLM", "true").lower() == "true"
    if gemini_key is None:
        gk = os.environ.get("GEMINI_API_KEY", "")
    else:
        gk = (gemini_key or "").strip() or os.environ.get("GEMINI_API_KEY", "")
    if openai_key is None:
        ok = os.environ.get("OPENAI_API_KEY", "")
    else:
        ok = (openai_key or "").strip() or os.environ.get("OPENAI_API_KEY", "")

    t0 = time.time()
    out_paths: list[Path] = []

    try:
        log.info("=" * 60)
        log.info("🚀 محرك مهووس للحسم v9.0 — بدء التشغيل")
        log.info("=" * 60)

        store_files = _load_csv_files(Path(store_dir))
        if not store_files:
            return EngineRunResult(False, error="لا توجد ملفات متجر CSV في المجلد المحدد.")

        log.info("📥 تحميل بيانات متجر مهووس...")
        store_df = load_store_products(store_files)
        if store_df.empty:
            return EngineRunResult(False, error="ملفات المتجر فارغة أو غير صالحة.")
        log.info(f"✅ {len(store_df):,} منتج في الجدار الواقي")

        brand_files = list(Path(brands_dir).glob("*.csv"))
        existing_brands = []
        if brand_files:
            existing_brands = load_brands(brand_files[0])
            log.info(f"✅ {len(existing_brands):,} ماركة محملة")
        else:
            log.info("ℹ️ لا يوجد ملف ماركات — سيتم الاستخراج تلقائياً")

        comp_files = _load_csv_files(Path(comp_dir))
        if not comp_files:
            return EngineRunResult(False, error="لا توجد ملفات منافسين CSV.")

        log.info("📦 تحميل بيانات المنافسين...")
        comp_df = load_competitor_products(comp_files)
        if comp_df.empty:
            return EngineRunResult(False, error="ملفات المنافسين فارغة.")
        log.info(f"✅ {len(comp_df):,} منتج من {len(comp_files)} منافس")
        for src in comp_df["source_file"].unique():
            count = len(comp_df[comp_df["source_file"] == src])
            log.info(f"   └── {Path(src).name}: {count:,} منتج")

        log.info("🧠 تحميل نموذج اللغة وبناء فهرس FAISS...")
        try:
            import torch
            torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        semantic_idx = SemanticIndex(model)
        semantic_idx.build(store_df, progress_cb=_log_cb)
        log.info(f"✅ FAISS جاهز: {len(store_df):,} متجه دلالي")

        oracle = None
        if use_llm:
            oracle = GeminiOracle(gemini_key=gk or "", openai_key=ok or "")
            if oracle.has_client:
                log.info("🤖 الذكاء الاصطناعي نشط للمنطقة الرمادية (Gemini و/أو OpenAI)")
            else:
                oracle = None
                log.info("ℹ️ لا يوجد مفتاح Gemini/OpenAI صالح — المنطقة الرمادية → مراجعة يدوية")
        else:
            log.info("ℹ️ الذكاء الاصطناعي غير مفعّل — المنطقة الرمادية → مراجعة يدوية")

        log.info(f"⚖️ بدء التحليل الهجين على {len(comp_df):,} منتج...")
        engine = MahwousEngine(
            semantic_index=semantic_idx,
            brands_list=existing_brands,
            gemini_oracle=oracle,
        )

        new_opps, duplicates, reviews, new_brands = engine.run(
            store_df=store_df,
            comp_df=comp_df,
            use_llm=use_llm and oracle is not None,
            progress_cb=_progress_cb,
            log_cb=_log_cb,
        )

        elapsed = time.time() - t0
        log.info("=" * 60)
        log.info(f"🎉 اكتمل التحليل في {elapsed:.1f} ثانية")
        log.info(f"   🌟 فرص جديدة:    {len(new_opps):,}")
        log.info(f"   🚫 مكررات:       {len(duplicates):,}")
        log.info(f"   🔍 مراجعة يدوية: {len(reviews):,}")
        log.info("=" * 60)

        import pandas as pd
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")

        def _results_to_df(results: list[MatchResult]) -> pd.DataFrame:
            return pd.DataFrame([{
                "اسم منتج المنافس":  r.comp_name,
                "صورة المنافس":      r.comp_image,
                "سعر المنافس":       r.comp_price,
                "مصدر الملف":        r.comp_source,
                "أقرب منتج لدينا":   r.store_name,
                "نسبة التطابق %":    f"{r.confidence*100:.1f}",
                "الطبقة المستخدمة":  r.layer_used,
                "الماركة":           r.brand,
                "القرار":            r.verdict,
                "وصف المنتج (LLM)":  r.generated_product_description,
                "وصف الماركة (LLM)": r.generated_brand_description,
            } for r in results])

        if new_opps:
            salla_bytes = export_salla_csv(new_opps)
            salla_path = output_dir / f"سلة_فرص_جديدة_{date_str}.csv"
            salla_path.write_bytes(salla_bytes)
            out_paths.append(salla_path)
            log.info(f"💾 ملف سلة: {salla_path.name}")

        if new_opps:
            df_new = _results_to_df(new_opps)
            p = output_dir / f"فرص_جديدة_{date_str}.csv"
            df_new.to_csv(p, index=False, encoding="utf-8-sig")
            out_paths.append(p)
            log.info(f"💾 فرص جديدة: {len(new_opps):,} منتج")

        if duplicates:
            df_dup = _results_to_df(duplicates)
            p = output_dir / f"مكررات_محظورة_{date_str}.csv"
            df_dup.to_csv(p, index=False, encoding="utf-8-sig")
            out_paths.append(p)
            log.info(f"💾 مكررات: {len(duplicates):,} منتج")

        if reviews:
            df_rev = _results_to_df(reviews)
            p = output_dir / f"مراجعة_يدوية_{date_str}.csv"
            df_rev.to_csv(p, index=False, encoding="utf-8-sig")
            out_paths.append(p)
            log.info(f"💾 مراجعة يدوية: {len(reviews):,} منتج")

        if new_brands:
            brands_bytes = export_brands_csv(new_brands)
            brands_path = output_dir / f"ماركات_جديدة_مطلوبة_{date_str}.csv"
            brands_path.write_bytes(brands_bytes)
            out_paths.append(brands_path)
            log.info(f"💾 ماركات جديدة: {len(new_brands):,} ماركة")

        summary_lines = [
            f"| المقياس | القيمة |",
            f"|:---|:---:|",
            f"| 📅 تاريخ التشغيل | {date_str} |",
            f"| 🏪 منتجات متجرنا | {len(store_df):,} |",
            f"| 🔍 منتجات المنافسين | {len(comp_df):,} |",
            f"| 🌟 **فرص جديدة** | **{len(new_opps):,}** |",
            f"| 🚫 مكررات محظورة | {len(duplicates):,} |",
            f"| 🔍 مراجعة يدوية | {len(reviews):,} |",
            f"| 🆕 ماركات جديدة | {len(new_brands):,} |",
            f"| ⏱️ وقت التشغيل | {elapsed:.1f} ثانية |",
            f"| 🤖 الذكاء الاصطناعي | {'نشط' if oracle else 'غير مفعّل'} |",
        ]
        summary_text = "\n".join(summary_lines)
        sp = output_dir / "summary.txt"
        sp.write_text(summary_text, encoding="utf-8")
        out_paths.append(sp)

        log.info("✅ تم حفظ كل النتائج في مجلد الإخراج")
        return EngineRunResult(
            success=True,
            output_paths=out_paths,
            summary_text=summary_text,
            elapsed_sec=elapsed,
            stats={
                "new_opps": len(new_opps),
                "duplicates": len(duplicates),
                "reviews": len(reviews),
                "new_brands": len(new_brands),
                "store": len(store_df),
                "competitors": len(comp_df),
            },
        )
    except Exception as e:
        log.exception("فشل التشغيل")
        return EngineRunResult(False, error=str(e), elapsed_sec=time.time() - t0)


def main():
    r = run_engine_paths(INPUT_STORE, INPUT_COMP, INPUT_BRANDS, OUTPUT_DIR)
    if not r.success:
        log.error(f"❌ {r.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
