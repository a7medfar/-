customer-churn-prediction/
├── data/                    # بيانات التدريب والاختبار
│   ├── raw/                # البيانات الأولية
│   ├── processed/          # البيانات المعالجة
│   └── external/           # بيانات خارجية
├── models/                 # النماذج المدربة
│   ├── churn_model.pkl    # النموذج الرئيسي
│   └── version_1/         # إصدارات سابقة
├── src/                   # الكود المصدري
│   ├── data_processing.py # معالجة البيانات
│   ├── feature_engineering.py # هندسة الميزات
│   ├── model_training.py  # تدريب النماذج
│   ├── api.py            # واجهة API
│   ├── monitoring.py     # نظام المراقبة
│   └── utils.py          # أدوات مساعدة
├── notebooks/            # دفاتر جupyter
│   ├── exploration.ipynb      # استكشاف البيانات
│   ├── model_training.ipynb   # تجريب النماذج
│   └── evaluation.ipynb       # تقييم الأداء
├── tests/               # الاختبارات
│   ├── test_data.py    # اختبارات البيانات
│   ├── test_model.py   # اختبارات النموذج
│   └── test_api.py     # اختبارات API
├── requirements.txt    # المتطلبات
├── Dockerfile         # إعداد Docker
├── docker-compose.yml # تكوين الحاويات
└── README.md          # هذا الملف
