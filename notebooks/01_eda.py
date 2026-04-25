import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv('data/creditcard.csv')
print(f"✅ Shape: {df.shape}")
print(f"✅ Fraud cases: {df['Class'].sum()}/{len(df)} = {df['Class'].mean():.4%}")

# Tạo reports folder
os.makedirs('reports', exist_ok=True)

# 1. PIE CHART FRAUD DISTRIBUTION
plt.figure(figsize=(10,6))
plt.pie([df['Class'].eq(0).sum(), df['Class'].eq(1).sum()], 
        labels=['Normal (99.83%)','Fraud (0.17%)'], 
        autopct='%1.2f%%', colors=['skyblue','red'], startangle=90)
plt.title('Credit Card Fraud Distribution - EXTREMELY IMBALANCED!', fontsize=14, fontweight='bold')
plt.savefig('reports/fraud_pie.png', dpi=300, bbox_inches='tight')
# plt.show()


# ===== 2. QUICK STATS =====
print("\n📊 DATA INSIGHTS:")
print(df['Class'].value_counts())
print("\nColumns:", df.columns.tolist())

# ===== 3. HISTOGRAMS FOR AMOUNT/TIME BY CLASS =====
print("\n⏳ Generating Histograms...")
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

sns.histplot(df[df['Class'] == 0]['Time'], bins=50, ax=ax[0, 0], color='skyblue')
ax[0, 0].set_title('Normal Transactions by Time')

sns.histplot(df[df['Class'] == 1]['Time'], bins=50, ax=ax[0, 1], color='red')
ax[0, 1].set_title('Fraud Transactions by Time')

sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, ax=ax[1, 0], color='skyblue')
ax[1, 0].set_title('Normal Transactions by Amount')
ax[1, 0].set_yscale('log')

sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, ax=ax[1, 1], color='red')
ax[1, 1].set_title('Fraud Transactions by Amount')
ax[1, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('reports/hist_time_amount.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ===== 4. CORRELATION HEATMAP (Top 10) =====
print("🔥 Generating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
corr = df.corr()
top_10_corr = corr['Class'].abs().nlargest(11).index
sns.heatmap(df[top_10_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Top 10 Feature Correlation with Class', fontsize=14, fontweight='bold')
plt.savefig('reports/corr_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== 5. BOXPLOTS =====
print("📦 Generating Boxplots...")
top_4_features = top_10_corr[1:5]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, feature in enumerate(top_4_features):
    sns.boxplot(x='Class', y=feature, data=df, ax=axes[i], hue='Class', palette={0: 'skyblue', 1: 'red'}, legend=False)
    axes[i].set_title(f'Boxplot of {feature} by Class')

plt.tight_layout()
plt.savefig('reports/boxplots.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("\n✅ Task EDA completed. Output saved: reports/hist_time_amount.png, reports/corr_heatmap.png, reports/boxplots.png.")