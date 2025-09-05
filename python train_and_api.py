# تثبيت المكتبات المطلوبة
!pip install mlflow xgboost fastapi uvicorn scikit-learn pandas numpy matplotlib nest_asyncio

# استيراد المكتبات
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
import os
from fastapi import FastAPI, HTTPException
import uvicorn
from contextlib import asynccontextmanager
import nest_asyncio
from threading import Thread
import asyncio

# تطبيق nest_asyncio للسماح بتشغيل asyncio في Jupyter
nest_asyncio.apply()

# تحميل البيانات
df = pd.read_json("C:/Users/DELL/Downloads/customer_churn_mini.json", lines=True)

print(" البيانات الأولية:")
print(df.head())
print(f"\n معلومات البيانات:")
print(df.info())

# تنظيف البيانات ومعالجة القيم المفقودة
df = df.dropna()

# تحويل الأعمدة الزمنية
df["time_stamp"] = pd.to_datetime(df["ts"], unit='ms')
df["registration_time"] = pd.to_datetime(df["registration"], unit='ms')

# تحويل أنواع البيانات - استخدام int بدلاً من category
# تأكد من أن القيم ليست NaN قبل التحويل
df['gender'] = df['gender'].map({'M': 0, 'F': 1}).astype('int')
df['level'] = df['level'].map({'free': 0, 'paid': 1}).astype('int')

# تحويل الأعمدة الأخرى إلى رقمية
categorical_cols = ['page', 'location', 'userAgent', 'auth', 'method']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes.astype('int')

df['status'] = df['status'].astype('int')

# التحقق من أنواع البيانات بعد التحويل
print("\n أنواع البيانات بعد التحويل:")
print(df.dtypes)

def create_features(df: pd.DataFrame):
    """
    إنشاء الميزات من البيانات الخام مع تحليل تلقائي churn
    """
    # نسخة من البيانات لحساب الفترات الزمنية
    df_temp = df.copy()
    df_temp['datetime'] = pd.to_datetime(df_temp['ts'], unit='ms')
    
    # ميزات المستخدم
    user_stats = df_temp.groupby('userId').agg(
        total_songs_played=('song', 'count'),
        distinct_artists_played=('artist', 'nunique'),
        avg_song_length=('length', 'mean'),
        subscription_length=('datetime', lambda x: (x.max() - x.min()).total_seconds() / (24 * 3600)),
        last_activity=('datetime', 'max')
    ).reset_index()

    # recency: الأيام منذ آخر نشاط
    current_time = pd.Timestamp.now()
    user_stats['activity_recency'] = (current_time - user_stats['last_activity']).dt.days
    user_stats.drop(columns=['last_activity'], inplace=True)

    # الحصول على الجندر والمستوى من البيانات الأصلية
    gender_level = df.groupby('userId')[['gender', 'level']].first().reset_index()
    user_stats = user_stats.merge(gender_level, on='userId', how='left')

    # تحليل توزيع النشاط لتحديد عتبة الـ churn المناسبة
    recency_stats = user_stats['activity_recency'].describe()
    print(" إحصائيات activity_recency:")
    print(recency_stats)
    
    # رسم توزيع النشاط
    plt.figure(figsize=(10, 6))
    plt.hist(user_stats['activity_recency'], bins=30, alpha=0.7, color='skyblue')
    plt.xlabel('Days since last activity')
    plt.ylabel('Number of users')
    plt.title('Distribution of user activity recency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # البحث عن عتبة مناسبة  churn
    thresholds = [7, 15, 30, 45, 60, 90]
    best_threshold = None
    best_balance = float('inf')
    
    print(" البحث عن أفضل عتبة  churn:")
    for threshold in thresholds:
        churn_ratio = (user_stats['activity_recency'] > threshold).mean()
        balance_score = abs(churn_ratio - 0.5)  # نحاول الوصول إلى توازن 50%
        
        print(f"   {threshold} أيام: نسبة churn = {churn_ratio:.2%}, توازن = {balance_score:.4f}")
        
        if balance_score < best_balance:
            best_threshold = threshold
            best_balance = balance_score
            best_ratio = churn_ratio
    
    print(f" أفضل عتبة: {best_threshold} أيام (نسبة churn: {best_ratio:.2%})")
    
    # إذا كانت جميع النسب 100%، نستخدم النسبة المئوية بدلاً من العتبة الثابتة
    if best_ratio == 1.0:
        print("  جميع المستخدمين مصنفين كـ churn، استخدام النسبة المئوية...")
        # استخدام أعلى 30% نشاطاً كـ non-churn
        threshold_percentile = user_stats['activity_recency'].quantile(0.3)
        user_stats['churn'] = (user_stats['activity_recency'] > threshold_percentile).astype('int')
        print(f" استخدام النسبة المئوية: الأعلى 30% نشاطاً (> {threshold_percentile:.1f} يوم) كـ non-churn")
    else:
        # تطبيق أفضل عتبة
        user_stats['churn'] = (user_stats['activity_recency'] > best_threshold).astype('int')

    # تنظيف البيانات النهائية
    user_stats = user_stats.dropna()
    
    # التحقق من توزيع الفئات
    churn_distribution = user_stats['churn'].value_counts()
    print(f" توزيع فئات الـ churn: {churn_distribution.to_dict()}")
    
    # إذا كانت لا تزال فئة واحدة، نخلق توازناً اصطناعياً
    if len(churn_distribution) == 1:
        print("  لا يزال هناك فئة واحدة، إنشاء توازن اصطناعي...")
        # نجعل النصف الأول non-churn والنصف الثاني churn
        user_stats = user_stats.sort_values('activity_recency')
        half_idx = len(user_stats) // 2
        user_stats.iloc[:half_idx, user_stats.columns.get_loc('churn')] = 0
        user_stats.iloc[half_idx:, user_stats.columns.get_loc('churn')] = 1
        
        churn_distribution = user_stats['churn'].value_counts()
        print(f" توزيع فئات الـ churn بعد التوازن الاصطناعي: {churn_distribution.to_dict()}")
    
    # التأكد من أن جميع الأعمدة رقمية
    for col in user_stats.columns:
        if user_stats[col].dtype.name == 'object' or user_stats[col].dtype.name == 'category':
            user_stats[col] = user_stats[col].astype('float64')
    
    # التحقق من أنواع البيانات النهائية
    print("\n أنواع البيانات في user_stats:")
    print(user_stats.dtypes)
    
    # رسم توزيع الفئات
    plt.figure(figsize=(8, 6))
    churn_distribution.plot(kind='bar', color=['green', 'red'])
    plt.title('Distribution of Churn Classes')
    plt.xlabel('Churn Status')
    plt.ylabel('Number of Users')
    plt.xticks([0, 1], ['Not Churn', 'Churn'], rotation=0)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return user_stats

def train_model(user_stats):
    """
    تدريب نموذج التنبؤ بانسحاب العملاء
    """
    feature_cols = ['total_songs_played', 'distinct_artists_played', 'avg_song_length',
                    'subscription_length', 'activity_recency', 'gender', 'level']
    
    # التأكد من أن جميع الأعمدة رقمية
    for col in feature_cols:
        if user_stats[col].dtype.name == 'object' or user_stats[col].dtype.name == 'category':
            user_stats[col] = user_stats[col].astype('float64')
    
    X = user_stats[feature_cols]
    y = user_stats['churn']

    # التحقق من توزيع الفئات
    print(f" توزيع الفئات في الهدف: {y.value_counts().to_dict()}")
    
    # إذا كانت هناك فئة واحدة فقط، لا يمكن تدريب النموذج
    if len(y.unique()) < 2:
        print(" تحذير: البيانات تحتوي على فئة واحدة فقط، لا يمكن تدريب النموذج")
        print(" الحل: تحقق من بياناتك أو غيّر تعريف الـ churn")
        return None, None, None

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f" حجم بيانات التدريب: {X_train.shape}")
    print(f" حجم بيانات الاختبار: {X_test.shape}")

    # التعامل مع عدم التوازن في البيانات
    if y_train.sum() > 0:
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        print(f"  معامل موازنة الفئات: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0
        print(" تحذير: لا توجد حالات churn في بيانات التدريب")

    # إنشاء وتدريب النموذج
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        n_estimators=100,
        random_state=42,
        enable_categorical=False  # تأكيد تعطيل الدعم التجريبي للفئات
    )
    
    # التحقق من أنواع البيانات قبل التدريب
    print("\n أنواع البيانات قبل التدريب:")
    print(X_train.dtypes)
    
    model.fit(X_train, y_train)

    # تقييم النموذج
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    # مصفوفة الارتباك
    cm = confusion_matrix(y_test, y_pred)
    
    print("=" * 50)
    print(" نتائج تدريب النموذج:")
    print("=" * 50)
    print(" تقرير التصنيف:\n", classification_report(y_test, y_pred))
    print(f" دقة النموذج: {accuracy:.4f}")
    print(f" ROC-AUC Score: {roc_auc:.4f}")
    print(f" مصفوفة الارتباك:\n{cm}")

    # رسم مصفوفة الارتباك
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not Churn', 'Churn'])
    plt.yticks(tick_marks, ['Not Churn', 'Churn'])
    
    # إضافة الأرقام إلى المربعات
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # حفظ النموذج
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/churn_model.pkl")
    print(" تم حفظ النموذج بنجاح!")

    # تسجيل مع MLflow
    try:
        mlflow.start_run()
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("churn_threshold", "auto")

        mlflow.log_metric("precision_0", report["0"]["precision"])
        mlflow.log_metric("recall_0", report["0"]["recall"])
        mlflow.log_metric("f1_0", report["0"]["f1-score"])
        mlflow.log_metric("precision_1", report["1"]["precision"])
        mlflow.log_metric("recall_1", report["1"]["recall"])
        mlflow.log_metric("f1_1", report["1"]["f1-score"])
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.xgboost.log_model(model, "model")
        mlflow.end_run()
        print(" تم تسجيل النموذج في MLflow بنجاح!")
    except Exception as e:
        print(f" تحذير: فشل التسجيل في MLflow: {e}")

    return model, X_test, y_test

# إعداد FastAPI مع lifespan handlers الحديثة
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    MODEL_PATH = "model/churn_model.pkl"
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(" تم تحميل النموذج بنجاح")
        else:
            print(" النموذج غير موجود، يرجى تدريبه أولاً")
            model = None
    except Exception as e:
        print(f" خطأ في تحميل النموذج: {e}")
        model = None
    yield
    # Shutdown
    print(" إيقاف الخادم")

app = FastAPI(
    title="Customer Churn Prediction API", 
    version="1.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """الصفحة الرئيسية"""
    return {
        "message": "Customer Churn Prediction API", 
        "status": "running",
        "endpoints": {
            "predict": "POST /predict",
            "model_info": "GET /model_info",
            "docs": "GET /docs"
        }
    }

@app.post("/predict")
async def predict(data: dict):
    """توقع انسحاب العميل"""
    if model is None:
        raise HTTPException(status_code=500, detail=" النموذج غير موجود. يرجى تدريب النموذج أولاً.")

    try:
        # تحويل البيانات إلى DataFrame
        feature_cols = ['total_songs_played', 'distinct_artists_played', 'avg_song_length',
                        'subscription_length', 'activity_recency', 'gender', 'level']
        
        # إنشاء DataFrame مع القيم الافتراضية للأعمدة المفقودة
        input_data = {}
        for col in feature_cols:
            input_data[col] = data.get(col, 0)
        
        df_input = pd.DataFrame([input_data])

        # التوقع
        prediction = model.predict(df_input)
        proba = model.predict_proba(df_input)[:, 1]

        return {
            "churn": int(prediction[0]),
            "probability": float(proba[0]),
            "status": "success",
            "message": "Churn prediction completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"فشل التوقع: {str(e)}")

@app.get("/model_info")
async def model_info():
    """معلومات عن النموذج"""
    if model is None:
        raise HTTPException(status_code=500, detail="النموذج غير متوفر")
    
    return {
        "model_type": "XGBoost",
        "features": ['total_songs_played', 'distinct_artists_played', 'avg_song_length',
                    'subscription_length', 'activity_recency', 'gender', 'level'],
        "status": "loaded" if model else "not_loaded",
        "n_features": 7
    }

# تشغيل الخادم في thread منفصل
def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())

# التشغيل الرئيسي
print("=" * 50)
print(" بدء معالجة البيانات...")
print("=" * 50)

# إنشاء الميزات
user_stats = create_features(df)
print(f" شكل البيانات بعد إنشاء الميزات: {user_stats.shape}")

# تدريب النموذج
print("=" * 50)
print(" بدء تدريب النموذج...")
print("=" * 50)

model, X_test, y_test = train_model(user_stats)

if model is not None:
    # اختبار التوقع
    print("=" * 50)
    print(" اختبار التوقع على عينة من البيانات:")
    print("=" * 50)
    
    if len(X_test) > 0:
        sample_data = X_test.iloc[0:1].to_dict('records')[0]
        print(f" بيانات العينة: {sample_data}")
        
        # توقع على العينة
        sample_pred = model.predict(pd.DataFrame([sample_data]))
        sample_proba = model.predict_proba(pd.DataFrame([sample_data]))[:, 1]
        
        print(f" التوقع: {'Churn' if sample_pred[0] == 1 else 'Not Churn'}")
        print(f" الاحتمالية: {sample_proba[0]:.4f}")
        
        # بدء الخادم في thread منفصل
        print("=" * 50)
        print(" بدء خادم FastAPI في الخلفية...")
        print(" يمكنك الوصول إلى الوثائق على: http://localhost:8000/docs")
        print("=" * 50)
        
        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # البقاء في الخلفية للسماح بالاختبار
        try:
            server_thread.join()
        except KeyboardInterrupt:
            print(" إيقاف الخادم")
    else:
        print(" لا توجد بيانات اختبار متاحة")
else:
    print(" فشل تدريب النموذج بسبب مشكلة في البيانات")
