# تقرير مشروع نظم استرجاع المعلومات

## 1. فكرة المشروع

المشروع هو نظام استرجاع معلومات Information Retrieval System. يقوم المستخدم بإدخال استعلام نصي، ثم يعيد النظام أفضل الوثائق المرتبطة بالاستعلام من مجموعة بيانات كبيرة. تم تنفيذ المشروع بلغة Python، وتجهيز واجهة Streamlit، وتقييم النظام باستخدام qrels ومقاييس IR القياسية.

الهدف العملي من النظام هو محاكاة محرك بحث صغير: يأخذ query، يعالجها، يطابقها مع الوثائق، يرتب النتائج، ثم يعرض الوثائق الأصلية للمستخدم.

## 2. Dataset

تم اختيار ClinicalTrials 2017 / TREC Precision Medicine 2017 لأنها تحقق شروط المشروع:

- ليست Antique dataset.
- تحتوي على أكثر من 200K وثيقة.
- تحتوي على queries.
- تحتوي على qrels، وهذا ضروري لحساب MAP و nDCG و Precision@10 و Recall.
- يمكن استخدامها كاملة بدون تجزيء Dataset ضخمة.

تم اعتماد Dataset واحدة بناءً على التوضيح النهائي للمعيدة، لذلك بُنيت جميع النماذج والتقييمات على هذه Dataset الكاملة.

النسخة المحضرة في المشروع تحتوي على:

- عدد الوثائق: 241,006
- Dataset: `clinicaltrials/2017/trec-pm-2017`
- عدد الاستعلامات: 30
- عدد qrels: 13,019
- مصدر البيانات: `ir_datasets`

تم التدريب وبناء الفهارس على Google Colab، ثم تم حفظ الملفات الناتجة داخل مجلد `artifacts` ونقلها إلى المشروع المحلي. وقت العرض لا يتم تدريب جديد، بل يقرأ النظام الملفات الجاهزة مثل `search_index.joblib` و `documents.sqlite`.

تم استخدام Dataset كاملة بدون أخذ أول N وثيقة فقط. هذا مهم لأن ملاحظات المعيدة أكدت ضرورة اختيار Dataset قابلة للمعالجة كاملة بدلاً من أخذ عينة من Dataset ضخمة.

## 3. المعالجة المسبقة

تم تنفيذ خدمة Preprocessing مستقلة تقوم بما يلي:

- تحويل النص إلى lowercase.
- استخراج الكلمات والأرقام فقط.
- إزالة stop words الإنجليزية.
- حذف الرموز والكلمات القصيرة جداً.

نستخدم نفس المعالجة للوثائق والاستعلامات لضمان التوافق بين تمثيل query وتمثيل documents.

عند بناء TF-IDF، يتم تمرير النص المنظف مسبقاً إلى `TfidfVectorizer` مع إيقاف tokenization الافتراضي داخله:

- `tokenizer=str.split`
- `preprocessor=None`
- `token_pattern=None`
- `lowercase=False`

وبذلك لا يحدث تنظيف مزدوج داخل TF-IDF.

بالنسبة إلى ClinicalTrials، تحتوي الوثيقة على عدة حقول مثل `title`, `condition`, `summary`, `detailed_description`, `eligibility`. لذلك يتم دمج كل الحقول النصية في نص واحد للفهرسة والبحث، مع حفظ النص الأصلي الكامل في قاعدة البيانات.

## 4. Query Processing و Query Refinement

تم تنفيذ Query Processing باستخدام نفس خطوات تنظيف الوثائق، ثم تمثيل الاستعلام بالطريقة المناسبة لكل نموذج: TF-IDF أو BM25 أو Embedding.

كما تمت إضافة Query Refinement قابل للتفعيل من الواجهة. هذه الخدمة تقوم بـ:

- تصحيح بعض الأخطاء الإملائية الشائعة مثل `diabetis` إلى `diabetes`.
- توسعة الاستعلام بمرادفات بسيطة مثل:
  - `treatment` إلى `therapy`, `medication`, `medicine`
  - `symptoms` إلى `signs`, `indications`
  - `diabetes` إلى `diabetic`, `glucose`, `insulin`

ويوجد خيار في الواجهة باسم `Query refinement` لتفعيل أو إيقاف هذه الميزة وتجربتها بشكل مستقل.

## 5. بنية النظام وفق SOA

تم تقسيم المشروع إلى خدمات Python مستقلة ومنظمة وفق مفهوم SOA، مع إضافة REST API باستخدام FastAPI كـ API Gateway. تستخدم واجهة Streamlit والـ API نفس `ServiceContainer` لتحميل قاعدة البيانات والفهرس وخدمات الاسترجاع وRAG مرة واحدة، مما يقلل التكرار ويحافظ على Loose Coupling.

الخدمات الأساسية:

- Data Service: تحميل وتجهيز Dataset.
- Document Store Service: تخزين الوثائق الأصلية في SQLite.
- Preprocessing Service: تنظيف النصوص.
- Query Refinement Service: تحسين الاستعلامات.
- Indexing Service: بناء فهارس TF-IDF وBM25 وEmbedding.
- Retrieval Service: تنفيذ البحث والترتيب.
- Evaluation Service: حساب MAP وnDCG وPrecision@10 وRecall.
- RAG Service: واجهة سؤال وجواب تعتمد على الوثائق المسترجعة.
- UI Service: واجهة Streamlit.
- API Gateway: واجهة REST توفر `/health` و`/search` و`/rag` و`/metrics`.

يمكن تشغيل واختبار الخدمات بشكل مستقل:

```powershell
.\run_api.cmd
.\run_tests.cmd
```

توفر FastAPI توثيق OpenAPI تفاعلياً على:

```text
http://127.0.0.1:8000/docs
```

```mermaid
flowchart LR
    A["ClinicalTrials Dataset"] --> B["Data Service"]
    B --> C["SQLite Store"]
    B --> D["Preprocessing"]
    D --> E["Indexing"]
    E --> F["TF-IDF"]
    E --> G["BM25"]
    E --> H["Embedding"]
    D --> Q["Query Refinement"]
    Q --> I["Retrieval"]
    F --> I
    G --> I
    H --> I
    I --> J["Evaluation"]
    I --> K["RAG Chat"]
    C --> K
    C --> L["UI"]
    I --> L
    I --> M["FastAPI Gateway"]
    K --> M
    C --> M
```

## 6. نماذج الاسترجاع

تم تنفيذ الطرق التالية:

- TF-IDF: تمثيل Vector Space Model وحساب cosine similarity.
- BM25: نموذج احتمالي مع إمكانية التحكم بالمعاملات k1 و b من الواجهة.
- Embedding: تمثيل latent semantic embedding باستخدام TF-IDF مع TruncatedSVD.
- Hybrid Parallel: دمج نتائج TF-IDF وBM25 وEmbedding باستخدام weighted score fusion.
- Hybrid Serial: استخدام BM25 لاسترجاع المرشحين ثم إعادة ترتيبهم باستخدام embedding.
- BERT Rerank: استخدام BM25 لجلب أفضل المرشحين بسرعة، ثم استخدام Sentence-BERT لإعادة ترتيب المرشحين دلالياً.

تم استخدام LSA / TruncatedSVD كـ embedding baseline سريع ومحلي، وتمت إضافة Sentence-BERT بطريقة reranking حتى لا نحسب BERT embeddings لكل 241,006 وثيقة. يجلب BM25 أفضل 50 مرشحاً، ثم يعيد BERT ترتيبهم دلالياً. تم تقييم BERT رسمياً، وكانت نتائجه أفضل بكثير من LSA، لذلك يمثل Sentence-BERT النموذج الدلالي الأساسي بينما بقي LSA للمقارنة العلمية.

## 7. الميزة الإضافية

لأن عدد أعضاء الفريق 5، المطلوب ميزة إضافية واحدة. تم اختيار RAG-style Chat.

تعمل الميزة بالشكل التالي:

1. يكتب المستخدم سؤالاً في تبويب RAG Chat.
2. يستخدم النظام Retrieval Service لجلب الوثائق الأكثر صلة.
3. تختار Grounded Answer Generator الجمل الأكثر ارتباطاً بكلمات السؤال من الوثائق المسترجعة.
4. يتم عرض مصادر الإجابة مع `doc_id` وscore.

الإجابة Extractive Grounded RAG تعمل محلياً دون API خارجي، وتضيف citations وتحتفظ بقائمة evidence حتى يمكن قياس groundedness.

## 8. التقييم

تم حساب المقاييس التالية:

- MAP@1000
- nDCG@10
- Precision@10
- Recall@1000

تم استرجاع أفضل 1000 وثيقة لكل query لحساب MAP وRecall بعمق عملي واضح، بينما بقيت مقاييس الرتب الأولى عند 10. النتائج محفوظة في `artifacts/evaluation_metrics.csv`:

| Method | MAP@1000 | nDCG@10 | Precision@10 | Recall@1000 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.0912 | 0.1259 | 0.1621 | 0.5902 |
| BM25 | 0.1921 | 0.2892 | 0.3000 | 0.6787 |
| LSA Embedding | 0.0013 | 0.0045 | 0.0138 | 0.0763 |
| Hybrid Parallel | 0.1476 | 0.2292 | 0.2345 | 0.6086 |
| Hybrid Serial | 0.1770 | 0.2948 | 0.2862 | 0.6468 |

حسب النتائج الفعلية، حقق BM25 أفضل MAP وPrecision@10 وRecall، بينما حقق Hybrid Serial أفضل nDCG@10. لذلك نستخدم BM25 وHybrid Serial/Parallel كطرق قوية في العرض، مع إبقاء BERT rerank كخيار دلالي إضافي فوق BM25.

تم أيضاً توليد تقييم إضافي بعد تفعيل Query Refinement في `artifacts/evaluation_metrics_refined.csv`:

| Method | MAP@1000 | nDCG@10 | Precision@10 | Recall@1000 |
|---|---:|---:|---:|---:|
| TF-IDF + Refinement | 0.0910 | 0.1350 | 0.1724 | 0.5794 |
| BM25 + Refinement | 0.1766 | 0.2742 | 0.3034 | 0.6494 |
| LSA + Refinement | 0.0019 | 0.0037 | 0.0103 | 0.1237 |
| Hybrid Parallel + Refinement | 0.1430 | 0.2324 | 0.2310 | 0.6007 |
| Hybrid Serial + Refinement | 0.1640 | 0.2786 | 0.2793 | 0.6390 |

كما تم توليد الرسوم البيانية في:

- `reports/figures/evaluation_metrics.png`
- `reports/figures/evaluation_metrics_refined.png`

أظهر Query Refinement تحسناً محدوداً في TF-IDF وLSA لبعض المقاييس، لكنه خفّض MAP وRecall في BM25 والهجين. السبب أن التوسعة بمرادفات عامة قد تضيف كلمات لا تناسب المصطلحات الطبية الدقيقة. لذلك جعلناها اختيارية من الواجهة بدلاً من فرضها دائماً.

### تقييم Sentence-BERT

| Method | MAP@10 | nDCG@10 | Precision@10 | Recall@10 |
|---|---:|---:|---:|---:|
| BERT Rerank | 0.0556 | 0.2474 | 0.2655 | 0.0864 |

تفوق BERT بوضوح على LSA في الرتب العشر الأولى. وهو أبطأ من BM25، لذلك استخدمناه كمرحلة reranking فوق 50 مرشحاً بدلاً من تطبيقه على كامل الوثائق في كل query.

### التقييم قبل وبعد RAG

قبل RAG يعرض النظام الوثائق المرتبة فقط. بعد RAG يضيف إجابة مؤرضة مع evidence وcitations دون تغيير قائمة الاسترجاع الأساسية. لذلك تم تقييم جودة طبقة RAG بالمقاييس التالية:

| Source P@5 | Source Recall@5 | Query Coverage | Citation Coverage | Groundedness |
|---:|---:|---:|---:|---:|
| 0.2483 | 0.0604 | 0.4352 | 0.6000 | 1.0000 |

قيمة Groundedness الكاملة تعني أن كل جملة دليل في الإجابة موجودة نصياً في الوثائق الأصلية المسترجعة من SQLite. Citation Coverage تساوي 0.6 لأن الإجابة تعرض أفضل ثلاثة أدلة من أصل خمسة مصادر مسترجعة.

## 9. لقطات من النظام

### Search Results

![Search Results](screenshots/search_results.png)

### RAG Answer

![RAG Answer](screenshots/rag_answer.png)

### RAG Sources

![RAG Sources](screenshots/rag_sources.png)

### Evaluation Chart

![Evaluation Chart](figures/evaluation_metrics.png)

## 10. الواجهة

تم بناء واجهة Streamlit تحتوي على:

- اختيار طريقة البحث.
- التحكم بمعاملات BM25.
- تفعيل أو إيقاف Query Refinement.
- اختيار BERT reranking من قائمة طرق البحث عند توفر النموذج.
- عرض أفضل 10 وثائق أصلية من SQLite.
- تبويب RAG Chat.
- تبويب Evaluation لعرض النتائج والرسم البياني.

## 11. تقسيم العمل بين أعضاء الفريق

- الخضر الديواني: تجهيز Dataset، إدارة qrels، وتحضير ملفات التدريب على Colab.
- نايا سعدون: Preprocessing و Query Processing و Query Refinement.
- حلا العوض: TF-IDF و Embedding وتمثيل الوثائق.
- ليث ضاهر: BM25 و Hybrid Retrieval و Ranking.
- نوال صالح: Evaluation و Streamlit UI و RAG Chat وكتابة التقرير.

## 12. طريقة التشغيل

وقت العرض لا نعيد التدريب. نشغل الواجهة فقط:

```powershell
cd "C:\Users\Lenovo\Desktop\ir dociment\ir_project"
.\run_app.cmd
```

ثم نفتح:

```text
http://localhost:8501
```

إذا أردنا إعادة بناء الفهارس من الصفر:

```powershell
$env:PYTHONPATH=".codex_deps;src"
python scripts\prepare.py --dataset clinicaltrials/2017/trec-pm-2017 --max-docs 0 --max-queries 0 --embedding-dims 64 --max-features 30000 --min-df 2 --max-df 0.95
python scripts\evaluate.py --dataset clinicaltrials/2017/trec-pm-2017 --max-queries 0
python scripts\evaluate_bert.py
python scripts\evaluate_rag.py
```

## 13. ميزة Documents Clustering

تمت إضافة خدمة مستقلة باسم `ClusteringService` لتنظيم نتائج البحث موضوعياً دون تغيير ترتيبها أو درجاتها. تعتمد الخدمة على تمثيلات LSA المحفوظة مسبقاً داخل `search_index.joblib`، ثم تطبق `MiniBatchKMeans` محلياً مرة واحدة على كامل الوثائق البالغ عددها 241,006 وثيقة.

تم اختيار 12 مجموعة موضوعية، وحُفظت النتائج في ملف مستقل باسم `artifacts/document_clusters.joblib`. لا تعدّل هذه العملية فهرس البحث أو قاعدة البيانات ولا تحتاج إلى إعادة تدريب على Colab. تعرض الواجهة خيار `Enable Document Clustering`، وعند تعطيله تبقى النتائج الأساسية كما هي، وعند تفعيله تُعرض النتائج ضمن مجموعات موضوعية مع المحافظة على `rank` و`score` الأصليين.

تم تقييم التجميع على عينة ثابتة من 2,500 وثيقة، وكانت النتائج:

| المقياس | القيمة |
|---|---:|
| عدد الوثائق | 241,006 |
| عدد المجموعات | 12 |
| Silhouette Score (Cosine) | 0.1411 |
| Davies–Bouldin Index | 3.0640 |
| Calinski–Harabasz Score | 67.8138 |

كما تم توليد رسم لتوزيع الوثائق بين المجموعات ورسم PCA لعينة من 5,000 وثيقة في:

- `reports/figures/clustering_distribution.png`
- `reports/figures/clustering_scatter.png`

لبناء ملف التجميع بشكل مستقل:

```powershell
python scripts\build_clusters.py --clusters 12
```

## 14. ميزة Offline Web Crawling

تمت إضافة عملية جمع مستقلة من المصدر الرسمي `ClinicalTrials.gov API v2`. يجمع السكربت 50 دراسة موزعة بالتساوي على السرطان والسكري وأمراض القلب والاكتئاب والجراحة، ثم يحفظ الاستجابات الخام كاملة في `data/crawled/crawled_trials_raw.json`، والنسخة المنظفة في `data/crawled/crawled_trials_clean.csv`، وإحصائيات الجمع في `data/crawled/crawling_metadata.json`.

تستخدم `CrawledRetrievalService` فهرس TF-IDF صغيراً يُبنى محلياً عند تشغيل المشروع من ملف CSV المحفوظ. لا تحتاج عملية البحث إلى الإنترنت، ولا تعدّل `search_index.joblib` أو `documents.sqlite`. يوفر خيار `Include Crawled Documents` مقارنة مستقلة: عند تعطيله تظهر نتائج Dataset الرسمية فقط، وعند تفعيله يظهر قسم منفصل باسم `Crawled Source Results` بعد النتائج الرسمية، ولا يتم خلط الدرجات أو الترتيب بين المصدرين.

نتائج الجمع والتقييم الفعلية:

| المقياس | القيمة |
|---|---:|
| الوثائق المطلوبة | 50 |
| الوثائق الناجحة | 50 |
| الوثائق الفاشلة | 0 |
| التكرارات المحذوفة | 0 |
| الوثائق الفريدة | 50 |
| متوسط طول الوثيقة | 107.04 كلمة |
| متوسط زمن البحث المحلي | 0.40 ms |
| أقصى زمن بحث في الاختبار | 1.06 ms |

لا تستخدم MAP أوnDCG أوPrecision أوRecall مع هذه الوثائق بسبب عدم توفر qrels لها. بدلاً من ذلك تم تقييم نجاح الجمع، إزالة التكرار، اكتمال البيانات، ومتوسط زمن البحث المحلي. الرسم محفوظ في `reports/figures/crawling_evaluation.png`.

```powershell
python scripts\crawl_clinicaltrials.py --per-query 10
python scripts\evaluate_crawled.py
```

## 15. ميزة Local Vector Store Retrieval

تمت إضافة طريقة استرجاع مستقلة باسم `vector_store` باستخدام مكتبة FAISS وإحداثيات LSA المحفوظة مسبقاً. حمّل سكربت `scripts/build_vector_store.py` المصفوفة الجاهزة من `search_index.joblib`، وطبّع 241,006 متجهات ذات 64 بعداً، ثم بنى `FAISS IndexFlatIP` وحفظه في `artifacts/vector_store.index` بحجم 58.84MB. لم تتم إعادة معالجة الوثائق أو إعادة تدريب SVD، ولم يُعدّل الفهرس الأساسي أو قاعدة البيانات.

تتحقق `VectorStoreService` من عدد الوثائق والأبعاد وdigest ترتيب `doc_ids` قبل تحميل الفهرس. يُحوّل الاستعلام بنفس TF-IDF وSVD وnormalization المستخدم في LSA، ثم يبحث باستخدام Inner Product الذي يعادل Cosine Similarity للمتجهات المطبّعة. تظهر الطريقة في قائمة Retrieval Method ويمكن استخدامها أيضاً عبر FastAPI وRAG.

تم تقييم FAISS رسمياً على 29 استعلاماً لها qrels وبعمق 1000. تطابقت جميع مقاييس الجودة ونسبة تداخل Top-10 مع LSA Brute Force، بينما انخفض متوسط زمن البحث بحوالي 49%:

| الطريقة | MAP@1000 | nDCG@10 | P@10 | Recall@1000 | Avg latency | Median | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSA Brute Force | 0.001263 | 0.004484 | 0.013793 | 0.076318 | 29.98 ms | 24.71 ms | 43.86 ms |
| FAISS Vector Store | 0.001263 | 0.004484 | 0.013793 | 0.076318 | 15.31 ms | 14.72 ms | 20.47 ms |

بلغ زمن تحميل FAISS Index نحو 158.85ms، وتطابقت نتائج Top-10 بنسبة 100%. تحفظ النتائج في `artifacts/vector_store_evaluation_metrics.csv` والرسم في `reports/figures/vector_store_evaluation.png`.

```powershell
python scripts\build_vector_store.py --backend faiss
python scripts\evaluate_vector_store.py
```

## 16. Feature Comparison: Before & After

تمت إضافة لوحة مقارنة في أعلى تبويب Evaluation لتوضيح وظيفة كل ميزة إضافية وطريقة تجربتها قبل وبعد التفعيل. لا تُعامل المقاييس المختلفة كأنها على مقياس واحد؛ بل يُعرض جدول موحد للشرح، ثم بطاقات رقمية ورسومات متخصصة لكل ميزة.

| Feature | OFF / Before | ON / After | Effect | Evaluation | How to Test |
|---|---|---|---|---|---|
| RAG Chat | نتائج وثائق فقط | جواب Grounded مع مصادر | تحويل نتائج الاسترجاع إلى جواب مفهوم | Groundedness / Citation Coverage | مقارنة Search مع RAG Chat للسؤال نفسه |
| Document Clustering | قائمة مرتبة عادية | نتائج مجمعة موضوعياً | تسهيل التصفح دون تغيير rank أوscore | Silhouette / Davies-Bouldin / Charts | تشغيل الاستعلام والزر OFF ثم ON |
| Offline Web Crawling | Dataset الرسمية فقط | قسم Crawled Source مستقل | إضافة مصدر خارجي محفوظ يعمل Offline | Success / Duplicates / Latency | تشغيل الاستعلام والزر OFF ثم ON |
| Local Vector Store | LSA Brute Force | FAISS Vector Store | الحفاظ على الجودة مع تقليل زمن البحث | Top-10 Overlap / Latency / MAP / nDCG | مقارنة embedding مع vector_store |

تقرأ البطاقات الأرقام مباشرة من ملفات CSV المحفوظة، وكانت القيم النهائية:

| البطاقة | القيمة |
|---|---:|
| RAG Groundedness | 100% |
| RAG Citation Coverage | 60% |
| Clustering Silhouette | 0.1411 |
| Davies-Bouldin | 3.0640 |
| Crawling Success | 50/50 |
| Crawled Offline Search Latency | 0.40 ms |
| FAISS Speed-up | 48.9% |
| FAISS Top-10 Overlap | 100% |

تم حفظ وصف المقارنة في `artifacts/feature_comparison.csv`. توضح اللوحة أن الإضافات لا تتداخل: RAG للإجابة، Clustering للتنظيم، Crawling لإضافة مصدر Offline، وVector Store لتسريع البحث المتجهي.

## 17. تصدير نتائج البحث إلى CSV

تمت إضافة `SearchExportService` وزر `Download Search Results as CSV` بعد عرض نتائج البحث. يحفظ النظام آخر عملية بحث في `session_state`، لذلك يبقى زر التحميل متاحاً ولا تختفي البيانات عند الضغط عليه. لا تغيّر عملية التصدير أي خوارزمية أو ترتيب أو score؛ بل تحول النتائج الظاهرة إلى ملف قابل للاستخدام في Excel أو التحليل اللاحق.

يحتوي الملف على: query، source، rank، doc_id، retrieval_method، score، title، condition، snippet، URL، cluster_id، cluster_label، إعداد Query Refinement، Top K، وقيم BM25. تحمل نتائج Dataset القيمة `official_dataset` في عمود source، وتحمل نتائج Crawling القيمة `crawled`. عند تفعيل Clustering تُملأ معلومات المجموعة للنتائج الرسمية فقط، وتبقى فارغة لنتائج Crawling.

يستخدم الملف UTF-8 BOM لضمان فتح النصوص بشكل صحيح في Excel، ويصدّر Snippet بطول محدود بدلاً من الوثيقة الكاملة. كما تُحمى الخلايا النصية التي تبدأ بعلامات صيغ Excel. تم اختبار دمج المصدرين، Metadata الخاصة بالـCluster، وبداية UTF-8 BOM ضمن الاختبارات الآلية.

## 18. سجل الاستعلامات Recent Queries

تمت إضافة تحسين واجهة بسيط يحفظ آخر خمسة استعلامات ناجحة داخل `st.session_state`. يظهر السجل مباشرة تحت خانة Search Query، ويمكن للمستخدم الضغط على أي استعلام سابق لإعادة تعبئة الحقل دون تنفيذ البحث تلقائياً. ينفذ المستخدم البحث بعد ذلك بالضغط على Search، مما يمنع تنفيذ استعلام غير مقصود.

تنظف `QueryHistoryService` المسافات، وتتجاهل الاستعلام الفارغ، وتمنع التكرار دون حساسية لحالة الأحرف. إذا تكرر استعلام يُنقل إلى بداية القائمة، وتبقى آخر خمسة عناصر فقط. يوفر زر `Clear History` مسح السجل، ويُفقد السجل تلقائياً عند انتهاء جلسة Streamlit لأنه لا يُحفظ في قاعدة البيانات. لا تؤثر هذه الإضافة على Ranking أوEvaluation أوخوارزميات الاسترجاع، وتُصنف كـUsability Improvement.

## 19. صناديق شرح طرق البحث

تمت إضافة Method Explanation Box أسفل قائمة Retrieval Method في الشريط الجانبي. يتغير النص تلقائياً عند اختيار TF-IDF أوBM25 أوLSA Embedding أوHybrid Parallel أوHybrid Serial أوBERT Re-ranking أوFAISS Vector Store، ويشرح باختصار آلية الاسترجاع والترتيب المستخدمة. يساعد هذا الصندوق في المناقشة والعرض ولا يغير أي خوارزمية أونتيجة.

عند تفعيل Query Refinement يظهر توضيح لعملية التصحيح والتوسعة بالمرادفات. وعند تفعيل Clustering يظهر تنبيه بأنه ينظم النتائج حسب الموضوع دون تغيير rank أوscore. وعند تفعيل Crawling يظهر تنبيه بأن النتائج الخارجية تعرض في قسم مستقل ولا تُخلط درجاتها مع ترتيب Dataset الرسمي. تصنف هذه الإضافة كتحسين Explainability وUsability للواجهة.

## 20. GitHub

تم تجهيز المشروع للرفع على GitHub مع تجاهل الملفات الضخمة مثل:

- `data/raw/`
- `artifacts/search_index.joblib`
- `artifacts/documents.sqlite`
- `artifacts/bert_model_cache`
- `.codex_deps`

الـ README يشرح طريقة تنزيل Dataset وتوليد الملفات الكبيرة محلياً أو على Colab.

## 21. المصادر

1. IR Datasets documentation: https://ir-datasets.com/
2. TREC Precision Medicine Track: https://trec.nist.gov/data/precmed.html
3. scikit-learn TF-IDF documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
4. scikit-learn TruncatedSVD documentation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
5. Reimers, N. and Gurevych, I. Sentence-BERT: https://arxiv.org/abs/1908.10084
6. Sentence Transformers documentation: https://www.sbert.net/
7. FastAPI documentation: https://fastapi.tiangolo.com/
8. Robertson, S. and Zaragoza, H. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 2009.
9. scikit-learn MiniBatchKMeans documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
10. ClinicalTrials.gov API v2: https://clinicaltrials.gov/data-api/api
11. FAISS documentation: https://faiss.ai/
