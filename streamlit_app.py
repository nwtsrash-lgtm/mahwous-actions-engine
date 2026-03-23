"""
واجهة محلية — رفع ملفات CSV وتشغيل محرك مهووس دون نسخها يدوياً إلى input/.
تشغيل: streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

# استقرار PyTorch/OpenMP على Windows — يُضبط قبل streamlit وأي مكتبة عددية
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# تقليل ذروة RAM أثناء الترميز (خطأ Windows 1455 / ملف ترحيل صغير)
os.environ.setdefault("MAHWOUS_ENCODE_BATCH", "32")
os.environ.setdefault("MAHWOUS_ST_BATCH", "4")

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# مسار المشروع
BASE = Path(__file__).resolve().parent
os.chdir(BASE)

from run_engine import OUTPUT_DIR, run_engine_paths


def _gemini_from_secrets() -> str:
    try:
        v = st.secrets.get("GEMINI_API_KEY", "")
        return str(v).strip() if v else ""
    except Exception:
        return ""


def _safe_name(name: str) -> str:
    return Path(name).name.replace("..", "_")


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        try:
            return pd.read_csv(pd.io.common.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    raise ValueError("تعذّر قراءة CSV. تأكد من الترميز.")


def _normalize_recheck_competitor_csv(src_file, dst_path: Path) -> None:
    """
    توحيد ملف الفرص الجديدة إلى بنية منافسين متوقعة:
    product_name, image_url, price, source_file
    """
    df = _read_csv_bytes(src_file.getvalue())
    df.columns = [str(c).strip() for c in df.columns]

    def _pick(cands: list[str]) -> str | None:
        low_map = {str(c).lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low_map:
                return low_map[c.lower()]
        for c in df.columns:
            lc = str(c).lower()
            if any(x in lc for x in cands):
                return c
        return None

    name_c = _pick(["اسم منتج المنافس", "product_name", "name", "اسم", "product", "title"])
    img_c = _pick(["صورة المنافس", "image_url", "image", "img", "صورة", "url", "src"])
    price_c = _pick(["سعر المنافس", "price", "سعر", "cost"])
    src_c = _pick(["مصدر الملف", "source_file", "source"])

    if name_c is None:
        raise ValueError("لم أجد عمود اسم المنتج داخل ملف الفرص الجديدة.")

    out = pd.DataFrame()
    out["product_name"] = df[name_c].fillna("").astype(str)
    out["image_url"] = df[img_c].fillna("").astype(str) if img_c else ""
    out["price"] = df[price_c].fillna("").astype(str) if price_c else ""
    out["source_file"] = df[src_c].fillna("recheck_uploaded_file.csv").astype(str) if src_c else "recheck_uploaded_file.csv"
    out = out[out["product_name"].str.strip() != ""].reset_index(drop=True)
    out.to_csv(dst_path, index=False, encoding="utf-8-sig")


def main():
    st.set_page_config(
        page_title="محرك مهووس للحسم",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🔬 محرك مهووس للحسم — تشغيل محلي")
    st.caption("ارفع ملفات المتجر والمنافسين (CSV) ثم اضغط تشغيل المحرك.")
    with st.expander("ظهر خطأ 1455 أو «ملف الترحيل صغير»؟"):
        st.markdown(
            "هذا يعني أن **RAM + الذاكرة الافتراضية** في Windows غير كافية أثناء تحميل النموذج.\n\n"
            "1. أغلق المتصفحات والبرامج الثقيلة ثم أعد المحاولة.\n"
            "2. زِد **ملف الترحيل**: إعدادات Windows ← حول ← إعدادات النظام المتقدمة ← الأداء ← "
            "إعدادات متقدمة ← الذاكرة الافتراضية ← **حجم مخصص** (مثلاً أولي 8192 ميجابايت، أقصى 16384).\n"
            "3. يمكن تقليل الحمل أكثر من PowerShell قبل التشغيل:  \n"
            "`$env:MAHWOUS_ENCODE_BATCH='16'; $env:MAHWOUS_ST_BATCH='2'`"
        )

    col1, col2 = st.columns(2)
    with col1:
        store_files = st.file_uploader(
            "ملفات متجر مهووس (CSV) — واحد أو أكثر",
            type=["csv"],
            accept_multiple_files=True,
            key="store",
        )
    with col2:
        comp_files = st.file_uploader(
            "ملفات منافسين (CSV) — واحد أو أكثر",
            type=["csv"],
            accept_multiple_files=True,
            key="comp",
        )

    brand_file = st.file_uploader(
        "ملف الماركات (CSV) — اختياري",
        type=["csv"],
        accept_multiple_files=False,
        key="brands",
    )

    use_llm = st.checkbox("تفعيل الذكاء الاصطناعي (LLM) للمنطقة الرمادية", value=False)
    api_g = st.text_input(
        "GEMINI_API_KEY (من Google AI Studio)",
        type="password",
        help="إن وُجد ملف .streamlit/secrets.toml يُستخدم تلقائياً عند ترك الحقل فارغاً",
    )
    api_o = st.text_input("OPENAI_API_KEY (بديل)", type="password", help="يُستخدم إن فشل Gemini أو لم يُضبط؛ أو OPENAI_API_KEY في البيئة")
    st.caption(
        "المطابقة الدلالية (Sentence Transformers + FAISS) تعمل دائماً بدون مفتاح. "
        "الـ LLM يُستخدم فقط لحسم المنتجات في «المنطقة الرمادية» (تقريباً بين 55% و88% تطابق)."
    )

    if st.button("▶ تشغيل المحرك", type="primary", use_container_width=True):
        if not store_files:
            st.error("أضف ملفاً واحداً على الأقل لمتجر مهووس.")
            return
        if not comp_files:
            st.error("أضف ملفاً واحداً على الأقل للمنافسين.")
            return

        with tempfile.TemporaryDirectory(prefix="mahwous_") as tmp:
            tmp = Path(tmp)
            d_store = tmp / "store"
            d_comp = tmp / "competitors"
            d_brands = tmp / "brands"
            d_store.mkdir()
            d_comp.mkdir()
            d_brands.mkdir()

            for f in store_files:
                (d_store / _safe_name(f.name)).write_bytes(f.getvalue())
            for f in comp_files:
                (d_comp / _safe_name(f.name)).write_bytes(f.getvalue())
            if brand_file is not None:
                (d_brands / _safe_name(brand_file.name)).write_bytes(brand_file.getvalue())

            out_dir = OUTPUT_DIR / "_ui_last"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)

            gk = api_g.strip() or _gemini_from_secrets() or None
            ok = api_o.strip() or None
            try:
                with st.spinner("جاري التشغيل… (تحميل النموذج وبناء FAISS قد يستغرق دقائق)"):
                    result = run_engine_paths(
                        d_store,
                        d_comp,
                        d_brands,
                        out_dir,
                        use_llm=use_llm,
                        gemini_key=gk,
                        openai_key=ok,
                    )
            except Exception as e:
                st.exception(e)
                return

        if not result.success:
            st.error(result.error)
            return

        st.success(
            f"اكتمل في {result.elapsed_sec:.1f} ث — فرص: {result.stats.get('new_opps', 0)} | "
            f"مكررات: {result.stats.get('duplicates', 0)} | مراجعة: {result.stats.get('reviews', 0)}"
        )
        st.markdown(result.summary_text)

        st.subheader("تحميل النتائج")
        cols = st.columns(3)
        for i, p in enumerate(result.output_paths):
            if not p.is_file():
                continue
            data = p.read_bytes()
            with cols[i % 3]:
                st.download_button(
                    label=f"⬇ {p.name}",
                    data=data,
                    file_name=p.name,
                    mime="text/csv" if p.suffix.lower() == ".csv" else "text/plain",
                    key=f"dl_{p.name}_{i}",
                )

        st.caption(f"نسخة محفوظة أيضاً في المجلد: `{out_dir}`")

    st.divider()
    st.subheader("🔁 إعادة التحقق من ملف الفرص الجديدة")
    st.caption("ارفع ملف الفرص الجديدة + ملف المتجر لإجراء فحص ثاني للنتائج.")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        recheck_new_file = st.file_uploader(
            "ملف الفرص الجديدة (CSV)",
            type=["csv"],
            accept_multiple_files=False,
            key="recheck_new_file",
        )
    with col_r2:
        recheck_store_file = st.file_uploader(
            "ملف المتجر (CSV)",
            type=["csv"],
            accept_multiple_files=False,
            key="recheck_store_file",
        )

    recheck_brand_file = st.file_uploader(
        "ملف الماركات (اختياري لإعادة التحقق)",
        type=["csv"],
        accept_multiple_files=False,
        key="recheck_brand_file",
    )
    recheck_use_llm = st.checkbox("تفعيل LLM في إعادة التحقق", value=False, key="recheck_use_llm")

    if st.button("🔎 تشغيل إعادة التحقق", use_container_width=True):
        if recheck_new_file is None or recheck_store_file is None:
            st.error("يلزم رفع ملف الفرص الجديدة وملف المتجر.")
            return

        with tempfile.TemporaryDirectory(prefix="mahwous_recheck_") as tmp:
            tmp = Path(tmp)
            d_store = tmp / "store"
            d_comp = tmp / "competitors"
            d_brands = tmp / "brands"
            d_store.mkdir()
            d_comp.mkdir()
            d_brands.mkdir()

            (d_store / _safe_name(recheck_store_file.name)).write_bytes(recheck_store_file.getvalue())
            _normalize_recheck_competitor_csv(recheck_new_file, d_comp / "recheck_competitors.csv")
            if recheck_brand_file is not None:
                (d_brands / _safe_name(recheck_brand_file.name)).write_bytes(recheck_brand_file.getvalue())

            out_dir = OUTPUT_DIR / "_ui_recheck_last"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)

            gk = api_g.strip() or _gemini_from_secrets() or None
            ok = api_o.strip() or None
            try:
                with st.spinner("جاري إعادة التحقق…"):
                    result2 = run_engine_paths(
                        d_store,
                        d_comp,
                        d_brands,
                        out_dir,
                        use_llm=recheck_use_llm,
                        gemini_key=gk,
                        openai_key=ok,
                    )
            except Exception as e:
                st.exception(e)
                return

        if not result2.success:
            st.error(result2.error)
            return

        st.success(
            f"إعادة التحقق اكتملت في {result2.elapsed_sec:.1f} ث — فرص: {result2.stats.get('new_opps', 0)} | "
            f"مكررات: {result2.stats.get('duplicates', 0)} | مراجعة: {result2.stats.get('reviews', 0)}"
        )
        st.markdown(result2.summary_text)

        st.subheader("تحميل نتائج إعادة التحقق")
        cols2 = st.columns(3)
        for i, p in enumerate(result2.output_paths):
            if not p.is_file():
                continue
            data = p.read_bytes()
            with cols2[i % 3]:
                st.download_button(
                    label=f"⬇ {p.name}",
                    data=data,
                    file_name=p.name,
                    mime="text/csv" if p.suffix.lower() == ".csv" else "text/plain",
                    key=f"dl_recheck_{p.name}_{i}",
                )
        st.caption(f"نسخة إعادة التحقق محفوظة في: `{out_dir}`")


if __name__ == "__main__":
    main()
