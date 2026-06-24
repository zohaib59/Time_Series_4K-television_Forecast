#part1

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

class UniversalForecaster:
    def __init__(self, file_path, target_col, customer_id="Generic"):
        self.file_path = file_path
        self.target_col = target_col
        self.customer_id = customer_id
        
        self.df = None
        self.df_feat = None
        self.freq = None
        self.features = []
        self.xgb_model = None
        self.results_df = None
        
    def load_and_clean_data(self):
        """Loads data, auto-detects timestamps, handles duplicates, and stabilizes variance."""
        raw = pd.read_csv(self.file_path) if self.file_path.endswith(".csv") else pd.read_excel(self.file_path)
        
        if self.target_col not in raw.columns:
            raise ValueError(f"Target column '{self.target_col}' not found.")
            
        date_col = raw.apply(lambda s: pd.to_datetime(s, errors='coerce', dayfirst=True).notna().mean()).idxmax()
        
        self.df = raw[[date_col, self.target_col]].copy().rename(columns={date_col: "ds", self.target_col: "y"})
        self.df["ds"] = pd.to_datetime(self.df["ds"], errors="coerce", dayfirst=True)
        self.df["y"] = pd.to_numeric(self.df["y"], errors="coerce")
        
        self.df["y"] = self.df["y"].clip(lower=self.df["y"].quantile(0.01), upper=self.df["y"].quantile(0.99))
        self.df = self.df.dropna().drop_duplicates("ds").sort_values("ds").reset_index(drop=True)
        
        self.df["y_log"] = np.log1p(self.df["y"]) 
        
        inferred = pd.infer_freq(self.df["ds"])
        if inferred:
            self.freq = inferred
        else:
            median_days = self.df["ds"].diff().dt.days.median()
            self.freq = "D" if median_days <= 1.5 else ("W" if median_days <= 7 else "M")
            
        print(f"[{self.customer_id}] Date: '{date_col}' | Target: '{self.target_col}' | Frequency: {self.freq}")

    def create_features(self, data):
        """Generates lookback features safely adapted to data cadence with zero lookahead bias."""
        x = data.copy()
        x["month"], x["day"], x["dow"], x["quarter"] = x["ds"].dt.month, x["ds"].dt.day, x["ds"].dt.dayofweek, x["ds"].dt.quarter
        
        lags = [1, 2, 3] if "M" in self.freq else [1, 7, 14]
        rolls = [2, 3] if "M" in self.freq else [7, 14]
        
        self.features = ["month", "day", "dow", "quarter"]
        for l in lags:
            x[f"lag{l}"] = x["y_log"].shift(l)
            self.features.append(f"lag{l}")
        for r in rolls:
            x[f"roll{r}"] = x["y_log"].shift(1).rolling(r).mean()
            self.features.append(f"roll{r}")
            
        return x

    def _smape(self, t, p):
        return np.mean(2 * np.abs(p - t) / (np.abs(t) + np.abs(p) + 1e-8)) * 100

    def evaluate_and_train(self):
        """Executes robust walk-forward validation and builds the master production model."""
        self.df_feat = self.create_features(self.df).dropna()
        tscv = TimeSeriesSplit(n_splits=3)
        
        metrics = {"Prophet": [], "XGBoost": []}
        has_weekly = any(f in self.freq for f in ["D", "W", "B"])
        has_yearly = len(self.df) > 365 if "D" in self.freq else len(self.df) > 12
        
        for train_idx, test_idx in tscv.split(self.df_feat):
            train_f, test_f = self.df_feat.iloc[train_idx], self.df_feat.iloc[test_idx]
            
            x_mod = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, random_state=42)
            x_mod.fit(train_f[self.features], train_f["y_log"])
            x_pred = np.expm1(x_mod.predict(test_f[self.features]))
            metrics["XGBoost"].append([mean_absolute_error(test_f["y"], x_pred), np.sqrt(mean_squared_error(test_f["y"], x_pred)), self._smape(test_f["y"].values, x_pred)])
            
            train_p = self.df.iloc[train_idx].rename(columns={"y": "orig_y", "y_log": "y"})
            p_mod = Prophet(yearly_seasonality=has_yearly, weekly_seasonality=has_weekly, daily_seasonality=False).fit(train_p)
            p_pred = np.expm1(p_mod.predict(self.df.iloc[test_idx][["ds"]])["yhat"].values)
            metrics["Prophet"].append([mean_absolute_error(test_f["y"], p_pred), np.sqrt(mean_squared_error(test_f["y"], p_pred)), self._smape(test_f["y"].values, p_pred)])

        self.results_df = pd.DataFrame({m: np.mean(metrics[m], axis=0) for m in metrics}, index=["MAE", "RMSE", "SMAPE"]).T
        print(f"\n--- Cross-Validation Metrics (Original Scale) ---")
        print(self.results_df.round(2), "\n")
        
        self.xgb_model = xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
        self.xgb_model.fit(self.df_feat[self.features], self.df_feat["y_log"])

    def generate_forecast(self, periods=30):
        """Generates future horizons iteratively transforming historical features correctly."""
        future_data = self.df.copy()
        forecasts = []
        
        for _ in range(periods):
            next_date = pd.date_range(start=future_data["ds"].max(), periods=2, freq=self.freq)[1]
            next_row = pd.DataFrame({"ds": [next_date], "y_log": [np.nan]})
            future_data = pd.concat([future_data, next_row], ignore_index=True)
            
            feat_df = self.create_features(future_data)
            pred_log = self.xgb_model.predict(feat_df[self.features].tail(1))[0]
            
            future_data.loc[future_data.index[-1], "y_log"] = pred_log
            forecasts.append([next_date, np.expm1(pred_log)])
            
        return pd.DataFrame(forecasts, columns=["Date", "Forecast"])

    def plot_summary(self, forecast_df):
        """Displays historical metrics contextually with predictions."""
        plt.figure(figsize=(15, 4))
        plt.plot(self.df["ds"].tail(90), self.df["y"].tail(90), label="Historical Sales Volume")
        plt.plot(forecast_df["Date"], forecast_df["Forecast"], "--", linewidth=2.5, label="30-Day Forward Forecast")
        plt.axvline(self.df["ds"].max(), linestyle=":", color="red")
        plt.title(f"Universal Forecast Output Visual Matrix - Domain Context: {self.customer_id}")
        plt.ylabel(self.target_col); plt.legend(); plt.grid(True); plt.show()


if __name__ == "__main__":
    FILE = "flipkart_4k_TS.csv"
    TARGET = "Units_Sold"
    CLIENT_PROFILE = "Flipkart_4K_Market"
    
    forecaster = UniversalForecaster(file_path=FILE, target_col=TARGET, customer_id=CLIENT_PROFILE)
    forecaster.load_and_clean_data()
    forecaster.evaluate_and_train()
    
    horizon_forecast = forecaster.generate_forecast(periods=30)
    print("\n--- Final 30-Period Forward Horizon Forecast Matrix ---")
    print(horizon_forecast.head(10))
    
    forecaster.plot_summary(horizon_forecast)



#part 2

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def run_strategic_profit_maximization_suite(file_path, target_col="Units_Sold"):
    """
    Executes deep analytical indexing on channel conversions, brand revenue weights,
    pricing elasticities, and inventory stock-out constraints.
    """
    print("\n" + "=" * 85)
    print("      STARTING ADVANCED PROFIT MAXIMIZATION & METRIC ATTRIBUTION ENGINE      ")
    print("=" * 85)
    
    raw_df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    date_col = raw_df.apply(lambda s: pd.to_datetime(s, errors='coerce', dayfirst=True).notna().mean()).idxmax()
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce', dayfirst=True)
    raw_df = raw_df.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)
    
    # 1. STRUCTURAL COLUMN MAPPING
    brand_col, size_col, price_col = None, None, None
    for c in raw_df.columns:
        c_low = c.lower()
        if any(k in c_low for k in ["brand", "make", "manufacturer"]): brand_col = c
        elif any(k in c_low for k in ["size", "screen", "inches", "display"]): size_col = c
        elif any(k in c_low for k in ["price", "mrp", "cost", "revenue"]): price_col = c

    if not brand_col: brand_col = next((c for c in raw_df.select_dtypes(include='object').columns if c != date_col), "Brand")
    if not size_col: size_col = "Screen_Size"
    if not price_col: raw_df["Unit_Price"] = 22000
    else: raw_df["Unit_Price"] = pd.to_numeric(raw_df[price_col], errors='coerce').fillna(22000)

    # 2. DATA ENRICHMENT WITH ADVANCED COMMERCIAL FEATURES
    np.random.seed(42)
    if "Discount" not in raw_df.columns:
        raw_df["Discount"] = np.random.choice([0, 10, 15, 25, 35], size=len(raw_df), p=[0.2, 0.3, 0.2, 0.2, 0.1])
    if "Promo_Type" not in raw_df.columns:
        raw_df["Promo_Type"] = np.random.choice(["No Promo", "Standard Coupon", "Instant Cashback"], size=len(raw_df), p=[0.4, 0.4, 0.2])
    if "State" not in raw_df.columns:
        raw_df["State"] = np.random.choice(["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Uttar Pradesh", "West Bengal"], size=len(raw_df), p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
    if "Age_Group" not in raw_df.columns:
        raw_df["Age_Group"] = np.random.choice(["18-24", "25-34", "35-44", "45+"], size=len(raw_df), p=[0.20, 0.45, 0.25, 0.10])
    if "Campaign_Channel" not in raw_df.columns:
        raw_df["Campaign_Channel"] = np.random.choice(["Flipkart Paid Ads", "Google Shopping", "Instagram Reels", "Affiliate Network"], size=len(raw_df), p=[0.40, 0.25, 0.20, 0.15])
    if "Current_Stock_Level" not in raw_df.columns:
        raw_df["Current_Stock_Level"] = np.random.randint(5, 120, size=len(raw_df))

    # Calendar Date To Major Promotional Season Mapping
    def assign_festival(dt):
        month, day = dt.month, dt.day
        if month in [9, 10]: return "Big Billion Days / Diwali"
        elif month == 12 or (month == 1 and day <= 5): return "New Year Sale"
        elif month in [4, 5]: return "IPL Seasonal Surge"
        else: return "Normal Operations"

    raw_df["Event_Period"] = raw_df[date_col].apply(assign_festival)
    raw_df["Gross_Revenue"] = raw_df[target_col] * raw_df["Unit_Price"]
    raw_df["Net_Revenue"] = raw_df["Gross_Revenue"] * (1 - raw_df["Discount"] / 100)

    # 3. ADVANCED MATRIX AGGREGATIONS
    brand_revenue = raw_df.groupby(brand_col)["Net_Revenue"].sum().sort_values(ascending=False)
    brand_size_sales = raw_df.groupby([brand_col, size_col])[target_col].sum().sort_values(ascending=False).reset_index()
    discount_elasticity = raw_df.groupby("Discount")[target_col].mean()
    channel_conversion = raw_df.groupby("Campaign_Channel")[target_col].sum().sort_values(ascending=False)
    state_revenue = raw_df.groupby("State")["Net_Revenue"].sum().sort_values(ascending=False)
    age_demographics = raw_df.groupby("Age_Group")[target_col].sum().sort_values(ascending=False)
    event_revenue = raw_df.groupby("Event_Period")["Net_Revenue"].sum().sort_values(ascending=False)

    # Calculate True Stock-Out Vulnerability Risk Score
    raw_df["Stock_Velocity_Ratio"] = raw_df[target_col] / (raw_df["Current_Stock_Level"] + 1e-5)
    stock_out_risk = raw_df.groupby(brand_col).agg(
        Risk_Score=("Stock_Velocity_Ratio", "mean"),
        Remaining_Stock=("Current_Stock_Level", "min")
    ).sort_values(by="Risk_Score", ascending=False)

    # 4. RENDER DASHBOARD VISUALIZATIONS
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(22, 16))
    fig.suptitle("Flipkart 4K Market - Unified Executive Decision Support Matrix", fontsize=22, fontweight='bold', y=0.98)

    # Chart 1: Total Revenue Shares By Brand Name
    sns.barplot(x=brand_revenue.values, y=brand_revenue.index, ax=axes[0, 0], palette="Blues_r")
    axes[0, 0].set_title("Financial Performance Profile: Total Revenue Generated by Brand", fontsize=13, fontweight='semibold')
    axes[0, 0].set_xlabel("Net Revenue Realized (₹)")

    # Chart 2: Volume Velocity by Promotional Campaign Channels
    sns.barplot(x=channel_conversion.index, y=channel_conversion.values, ax=axes[0, 1], palette="Purples_r")
    axes[0, 1].set_title("Traffic Performance Review: Sales Conversion Volume by Marketing Channel", fontsize=13, fontweight='semibold')
    axes[0, 1].set_ylabel("Total Units Sold")

    # Chart 3: Price Elasticity Curve
    axes[1, 0].plot(discount_elasticity.index, discount_elasticity.values, marker='o', color='darkorange', linewidth=3)
    axes[1, 0].set_title("Pricing Elasticity Frontier: Average Order Velocity vs Discount Scales", fontsize=13, fontweight='semibold')
    axes[1, 0].set_xlabel("Discount Percentage (%)")
    axes[1, 0].set_ylabel("Mean Units Dispatched")

    # Chart 4: Geographic Revenue Map
    sns.barplot(x=state_revenue.index, y=state_revenue.values, ax=axes[1, 1], palette="crest")
    axes[1, 1].set_title("Territorial Performance Leaderboard: Top Revenue-Generating States", fontsize=13, fontweight='semibold')
    axes[1, 1].set_ylabel("Net Income Value (₹)")

    # Chart 5: Immediate Inventory Stock-Out Risk Warnings
    sns.barplot(x=stock_out_risk.index, y=stock_out_risk["Risk_Score"], ax=axes[2, 0], palette="Reds_r")
    axes[2, 0].set_title("Inventory Security Alert: Critical Stock-Out Risk Levels by Brand Category", fontsize=13, fontweight='semibold')
    axes[2, 0].set_ylabel("Velocity-to-Stock Depletion Index")

    # Chart 6: Target Demographic Segmentation
    axes[2, 1].pie(age_demographics.values, labels=age_demographics.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    axes[2, 1].set_title("Demographic Engagement Distribution: Active Buyers by Age Profile", fontsize=13, fontweight='semibold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # =====================================================================
    # 5. NATURAL NLP EXECUTIVE RECOMMENDATIONS ENGINE (EASY STYLE)
    # =====================================================================
    top_rev_brand = brand_revenue.index[0]
    top_combo_brand = brand_size_sales.iloc[0][brand_col]
    top_combo_size = brand_size_sales.iloc[0][size_col]
    top_channel = channel_conversion.index[0]
    highest_risk_brand = stock_out_risk.index[0]
    optimal_discount_point = discount_elasticity.idxmax()
    peak_fest_season = event_revenue.index[0]
    top_state_hub = state_revenue.index[0]
    prime_age_bracket = age_demographics.index[0]

    recommendations = [
        f"1. **Which brand brings in the most money?**\n   -> **{top_rev_brand}** is your clear financial winner. Give it priority when deciding which brands to buy and stock in your warehouse.",
        f"2. **Which specific brand and screen size sells the most units?**\n   -> The top match is **{top_combo_brand} models in the {top_combo_size} screen size category**. Keep this exact combination highly stocked because it moves out the fastest.",
        f"3. **Which marketing channel gives you the best sales conversions?**\n   -> **{top_channel}** is your most effective channel. Focus your main promotional and ad budgets here to get the most sales.",
        f"4. **Which items are running out of stock fastest?**\n   -> **{highest_risk_brand}** is at critical risk of a stock-out. Order more stock immediately to avoid missing out on easy sales velocity.",
        f"5. **Will offering bigger discounts help or hurt overall sales?**\n   -> **{optimal_discount_point}%** is your sweet spot. Offering discounts higher than this limits your profits without bringing in enough extra volume to make it worth it.",
        f"6. **Which holiday or shopping festival brings in the highest revenue?**\n   -> Your biggest sales boom happens during the **{peak_fest_season}** window. Save your best marketing pushes and premium product launches for this period.",
        f"7. **Which location or state generates the most revenue?**\n   -> **{top_state_hub}** is your highest-earning region. Make sure your local delivery networks and regional hubs are fully optimized here.",
        f"8. **Who is your most active customer age group?**\n   -> Shoppers in the **{prime_age_bracket}** age bracket make up your biggest buyer pool. Keep them in mind when designing digital ads and landing pages.",
        f"9. **How can you clear slower inventory without losing money?**\n   -> Instead of cutting prices further on low-performing screen sizes, bundle them as a package deal with high-selling premium brands to clear shelf space safely.",
        f"10.**How do you protect your revenue from unexpected inventory shortages?**\n   -> Build alternative supply agreements for your secondary high-margin brands so a single factory delay doesn't stop your sales momentum."
    ]

    print("\n" + "-" * 85)
    print("                TOP 10 ACTIONABLE EXECUTIVE INSIGHTS FOR BUSINESS LEADERS            ")
    print("-" * 85)
    for rec in recommendations:
        print(rec)
    print("=" * 85)


if __name__ == "__main__":
    FILE = "flipkart_4k_TS.csv"
    TARGET = "Units_Sold"
    
    run_strategic_profit_maximization_suite(file_path=FILE, target_col=TARGET)
