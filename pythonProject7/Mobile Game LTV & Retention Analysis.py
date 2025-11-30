#PROJECT: Mobile Game LTV & Retention Analysis

##Business Problem

#Bu projede amaç, mobil oyundaki kullanıcıların gelecekte yapacağı harcamaları (LTV)
#tahmin etmek ve gelir artırıcı içgörüler üretmektir.

#Bu doğrultuda şu sorulara yanıt aranır:
#Hangi kullanıcılar yüksek harcama potansiyeline sahip?
#Harcamayı en çok etkileyen özellikler neler?
#Kullanıcı davranışı (oturum, süre, ülke, cihaz vb.) LTV'yi nasıl şekillendiriyor?

#Data Story
#Veri seti, bir mobil oyunun kullanıcılarına ait davranış ve harcama bilgilerini içerir.
#Her satır bir oyuncuyu temsil eder.
#Veri; demografi, cihaz, oyun alışkanlıkları, oturum detayları, ödeme davranışları ve toplam harcama bilgilerini kapsar.
#Bu verilerle:
#Kullanıcı segmentasyonu yapılabilir,
#Harcamayı etkileyen faktörler analiz edilebilir,
#LTV tahmin modelleri kurulabilir,
#Anormal veya şüpheli kullanıcı davranışları belirlenebilir.


import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
sns.set(style="whitegrid")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#############################################
# 1. GENEL RESİM
#############################################
df = pd.read_csv("mobile_game_inapp_purchases.csv")
df.head()
df.info()
df.describe(include="all")



def check_dataframe(df, head=5):
    print("### Shape ###")
    print(df.shape)

    print("\n### Types ###")
    print(df.dtypes)

    print("\n### Head ###")
    print(df.head(head))

    print("\n### Tail ###")
    print(df.tail(head))

    print("\n### Missing Values ###")
    print(df.isnull().sum())

    print("\n### Describe ###")
    print(df.describe().T)





#############################################
# 2. KATEGORİK DEĞİŞKEN ANALİZİ
#############################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    #print("##################### Tail #####################")
    #print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car



cat_cols, num_cols, cat_but_car = grab_col_names(df)

#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################

def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)




#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)




#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################
## Hedef değişkenin ortalamaya göre çıktısı

for col in num_cols:
    num_summary(df, col, plot=True)


def target_summary_with_cat(dataframe, target, categorical_col):

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "InAppPurchaseAmount", col)


##Hedef değişkenimizin ;
#Her kategorinin ortalama harcaması
#Kategori sayıları
#Oranlar
#Sıralı şekilde görünüm
#Boş değer kontrolü    ;

def target_summary_with_cat(dataframe, target, categorical_col):
    summary_df = pd.DataFrame({
        "COUNT": dataframe[categorical_col].value_counts(),
        "RATIO": dataframe[categorical_col].value_counts() / len(dataframe),
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()
    })

    summary_df = summary_df.sort_values("TARGET_MEAN", ascending=False)

    print(f"##### {categorical_col} #####")
    print(summary_df, end="\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "InAppPurchaseAmount", col)



#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################

def find_correlation(dataframe, numeric_cols, target="InAppPurchaseAmount", corr_limit=0.60):
    high_correlations = []
    low_correlations = []

    for col in numeric_cols:
        if col == target:
            continue  # Hedef değişkenin kendisini atlıyoruz

        corr_value = dataframe[[col, target]].corr(method="spearman").loc[col, target]

        print(f"{col}: {corr_value}")

        if abs(corr_value) > corr_limit:
            high_correlations.append(f"{col}: {corr_value}")
        else:
            low_correlations.append(f"{col}: {corr_value}")

    return low_correlations, high_correlations

low_corrs, high_corrs = find_correlation(df, num_cols)


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Outliers (Aykırı Değerler)
# 2. Missing Values (Eksik Değerler)
# 3. Feature Extraction (Özellik Çıkarımı)
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling (Özellik Ölçeklendirme)



#############################################
# 1. Outliers (Aykırı Değerler)
#############################################
sns.boxplot(x=df["InAppPurchaseAmount"], data=df)
plt.show(block=True)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)



#############################################
# 2. Missing Values (Eksik Değer Analizi ve Doldurma)
#############################################

# Güvenli Kopya
df1 = df.copy()
print("df1 kopyası başarıyla oluşturuldu.")
print(df1.head())

def missing_values_table(dataframe, na_name=False):
    """
    Veri setindeki eksik değerleri özetler.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (n_miss / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["Missing Values", "Ratio (%)"])

    print("\n--- Missing Value Summary ---\n")
    print(missing_df)

    if na_name:
        return na_columns


def fill_missing_values(dataframe, num_cols, cat_cols):
    """
    Eksik değerleri sayısal ve kategorik değişkenlere göre doldurur.
    """
    print("\n--- Filling Missing Values ---\n")

    # Sayısal değişkenler: median
    for col in num_cols:
        if dataframe[col].isnull().sum() > 0:
            dataframe[col] = dataframe[col].fillna(dataframe[col].median())
            print(f"{col} → median ile dolduruldu.")

    # Kategorik değişkenler: mode
    for col in cat_cols:
        if dataframe[col].isnull().sum() > 0:
            dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
            print(f"{col} → mode ile dolduruldu.")

    print("\nEksik değer doldurma işlemi tamamlandı.\n")
    return dataframe


#############################################
# 2.1 Missing Value Analysis
#############################################

print("Eksik değer analizi başlatılıyor...")
missing_values_table(df1)

#############################################
# 2.2 Eksik Değer Doldurma
#############################################

df1 = fill_missing_values(df1, num_cols=num_cols, cat_cols=cat_cols)

print("\ndf1 için eksik değer doldurma işlemi tamamlandı. Artık df1 temiz!")


#############################################
# 3. Feature Extraction (Özellik Çıkarımı)
#############################################

def feature_extraction(df):
    df = df.copy()
    print("--- Feature Extraction Started ---")

    # -----------------------------------
    # 1. Engagement Features
    # -----------------------------------
    df["SessionsPerDay"] = df["SessionCount"] / (df["FirstPurchaseDaysAfterInstall"] + 1)

    df["SessionLengthCategory"] = pd.cut(
        df["AverageSessionLength"],
        bins=[0, 10, 20, 60],
        labels=["Short", "Medium", "Long"]
    )

    df["IsHeavySessionUser"] = (df["AverageSessionLength"] > df["AverageSessionLength"].median()).astype(int)

    # -----------------------------------
    # 2. Purchase Behaviour Features
    # -----------------------------------
    df["PurchaseFrequency"] = df["InAppPurchaseAmount"] / (df["FirstPurchaseDaysAfterInstall"] + 1)

    df["IsHighSpender"] = (df["InAppPurchaseAmount"] > df["InAppPurchaseAmount"].median()).astype(int)

    df["PurchaseDelayGroup"] = pd.cut(
        df["FirstPurchaseDaysAfterInstall"],
        bins=[-1, 7, 30, 180],
        labels=["Early", "Medium", "Late"]
    )

    # -----------------------------------
    # 3. Demographic Features
    # -----------------------------------
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 18, 35, 60, 100],
        labels=["Teen", "YoungAdult", "Adult", "Senior"]
    )

    region_map = {
        "Turkey":"EMEA", "France":"EMEA", "Germany":"EMEA", "Spain":"EMEA", "Italy":"EMEA",
        "Norway":"EMEA", "Sweden":"EMEA", "Egypt":"EMEA", "Iran":"EMEA", "India":"APAC",
        "China":"APAC", "Japan":"APAC", "Brazil":"LATAM", "Mexico":"LATAM", "USA":"NA",
        "Canada":"NA", "Australia":"APAC", "Russia":"EMEA", "UK":"EMEA", "Switzerland":"EMEA"
    }
    df["CountryRegion"] = df["Country"].map(region_map).fillna("Other")

    print("--- Feature Extraction Completed Successfully ---")
    return df

# Uygulama
df2 = feature_extraction(df1)

print("Yeni feature'lar oluşturuldu! df2 hazır.")
df2.head()
df2.info()
df2.describe().T
df2[[ "SessionsPerDay","SessionLengthCategory","IsHeavySessionUser","PurchaseFrequency","IsHighSpender","PurchaseDelayGroup","AgeGroup","CountryRegion"]].head(10)


#############################################
# 4. One-Hot Encoding
#############################################

cat_cols, num_cols, cat_but_car = grab_col_names(df2)


def one_hot_encoder(df, categorical_cols, drop_first=True):
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded


df2_encoded = one_hot_encoder(df2, cat_cols, drop_first=True)
df2_encoded.head()


#############################################
# 5. Feature Scaling (Özellik Ölçeklendirme)
#############################################


# 1. Sayısal değişkenleri belirleme
cat_cols, num_cols, cat_but_car = grab_col_names(df2_encoded)

# Hedef değişkeni ölçekleme dışında
num_cols = [col for col in num_cols if col not in ["InAppPurchaseAmount"]]

# 2. StandardScaler ile ölçeklendirme
scaler = StandardScaler()
df2_encoded[num_cols] = scaler.fit_transform(df2_encoded[num_cols])

# 3. Fonksiyon haline getirme (opsiyonel, clean code)
def feature_scaling(dataframe, num_cols, target_col="InAppPurchaseAmount"):
    """
    Sayısal özellikleri ölçeklendirir, hedef değişkeni bırakır.
    """
    scaler = StandardScaler()
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if col not in [target_col]]
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe

# Uygulama
df2_encoded = feature_scaling(df2_encoded, num_cols)

# 4. Korelasyon Analizi ve Görselleştirme

numeric_df = df2_encoded.select_dtypes(include=['int32', 'int64', 'float64'])

plt.figure(figsize=(25, 12))
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show(block=True)


#############################################
#   NA TEMİZLEME
#############################################

# Target’ta (InAppPurchaseAmount) NaN varsa sil
df2_encoded = df2_encoded.dropna(subset=["InAppPurchaseAmount"])

# Sayısal kolonları median ile doldurma
numeric_cols = df2_encoded.select_dtypes(include=["int64", "float64"]).columns
df2_encoded[numeric_cols] = df2_encoded[numeric_cols].fillna(df2_encoded[numeric_cols].median())

# Kategorik kolonları mode ile doldurma
categorical_cols = df2_encoded.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    df2_encoded[col] = df2_encoded[col].fillna(df2_encoded[col].mode()[0])

df2_encoded = df2_encoded.dropna(subset=["InAppPurchaseAmount"])




# -----------------------------
# MODEL PIPELINE
#-----------------------------

TARGET = "InAppPurchaseAmount"
HANDLE_TARGET_NANS = "drop"   # "drop" veya "fill_zero"

# -----------------------------
# 1) Model için dataframe hazırla (NUMERIC only)
# -----------------------------
# df2_encoded: encoding & scaling yapılmış son dataframe

data = df2_encoded.copy()

# 1.a Target NaN handling
if HANDLE_TARGET_NANS == "drop":
    before = data.shape[0]
    data = data.dropna(subset=[TARGET])
    after = data.shape[0]
    print(f"Target NaN removal -> satır: {before} -> {after} (silindi: {before-after})")
elif HANDLE_TARGET_NANS == "fill_zero":
    n_missing = data[TARGET].isna().sum()
    data[TARGET] = data[TARGET].fillna(0)
    print(f"Target NaN fill_zero -> doldurulan satır sayısı: {n_missing}")
else:
    raise ValueError("HANDLE_TARGET_NANS must be 'drop' or 'fill_zero'")

# 1.b X,y ayırma ve sadece numerikler
# Select numeric columns (int/float)
numeric_df = data.select_dtypes(include=[np.number]).copy()

# Ensure target exists in numeric_df; if not, bring it (should be numeric)
if TARGET not in numeric_df.columns and TARGET in data.columns:
    numeric_df[TARGET] = data[TARGET]

print("Numeric DF shape:", numeric_df.shape)
print("Numeric DF columns tail:\n", numeric_df.columns[-20:])

# 1.c Son bir eksik değer kontrolü ve doldurma (güvenlik amaçlı)
missing_counts = numeric_df.isnull().sum()
if missing_counts.sum() > 0:
    print("Numeric kolonlarda eksikler var. Aşağıda listeleniyor:")
    print(missing_counts[missing_counts > 0])
    # Basitçe median ile dolduralım (model için güvenli)
    numeric_df = numeric_df.fillna(numeric_df.median())
    print("Eksikler median ile dolduruldu.")
else:
    print("Numeric kolonlarda eksik yok.")

# 1.d Son X,y
X = numeric_df.drop(columns=[TARGET])
y = numeric_df[TARGET]

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)
print("Hedef özet:\n", y.describe())

# -----------------------------
# 2) Baseline Linear Regression (quick check)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_test = lr.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nBaseline LinearRegression -> TEST RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

# quick scatter
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred_test, s=15)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"LR Test (RMSE={test_rmse:.2f})")
plt.show(block=True)

# -----------------------------
# 3) Compare a few models (CV) - lightweight
# -----------------------------
models = [
    ("LinearReg", LinearRegression()),
    ("RF", RandomForestRegressor(n_estimators=100, random_state=42))
]

print("\nCross-validated RMSE (5-fold):")
for name, m in models:
    rmse_cv = np.mean(np.sqrt(-cross_val_score(m, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"{name}: {rmse_cv:.4f}")

# -----------------------------
# 4) GridSearch on RandomForest (smaller grid to save time)
# -----------------------------
rf = RandomForestRegressor(random_state=42)
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 8],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print("\nGridSearch best params:", grid.best_params_)

# Evaluate on test
y_pred_rf = best_rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)
print(f"RF TEST RMSE: {rf_rmse:.4f}, R2: {rf_r2:.4f}")

# Feature importance (if available)
if hasattr(best_rf, "feature_importances_"):
    fi = pd.DataFrame({"feature": X.columns, "importance": best_rf.feature_importances_}).sort_values("importance", ascending=False).head(30)
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", data=fi)
    plt.title("RF Top Feature Importances")
    plt.tight_layout()
    plt.show(block=True)
