# ===== 01_FULL_EDA.py - 6 CHARTS FOR WORD =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('reports', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

df = pd.read_csv('data/creditcard.csv')
print(f"Dataset: {df.shape} | Fraud: {df['Class'].sum()} ({df['Class'].mean():.3%})")

# CHART 1: FRAUD PIE (CŨ)
plt.figure(figsize=(8,6))
plt.pie([df['Class'].eq(0).sum(), df['Class'].sum()], labels=['Bình thường','Gian lận'], 
        autopct='%1.2f%%', colors=['lightblue','coral'])
plt.title('Hình 1: Tỷ lệ gian lận thẻ (0.17%)')
plt.ylabel('')
plt.savefig('reports/01_fraud_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# CHART 2: AMOUNT HIST + BOX (CŨ)
fig, axes = plt.subplots(1, 2, figsize=(12,4))
df[df['Class']==0]['Amount'].hist(bins=50, ax=axes[0], color='blue', alpha=0.6)
axes[0].set_title('Số tiền giao dịch bình thường')
df[df['Class']==1]['Amount'].hist(bins=20, ax=axes[1], color='red', alpha=0.6)
axes[1].set_title('Số tiền gian lận')
plt.tight_layout()
plt.savefig('reports/02_amount_hist.png', dpi=300, bbox_inches='tight')
plt.close()

# CHART 3: TIME BY HOUR
df['hour'] = (df['Time'] % (24*3600)) / 3600  # Giờ trong ngày
fig, ax = plt.subplots(figsize=(12,6))
bins = np.arange(0, 49, 2)
df[df['Class']==0]['hour'].hist(bins=bins, ax=ax, alpha=0.6, label='Bình thường', color='green')
df[df['Class']==1]['hour'].hist(bins=bins, ax=ax, alpha=0.8, label='Gian lận', color='red')
ax.set_title('Hình 3: Thời gian giao dịch theo giờ')
ax.set_xlabel('Giờ (0-48h)')
ax.legend()
plt.savefig('reports/03_time_hour.png', dpi=300, bbox_inches='tight')
plt.close()

# CHART 4: CORRELATION TOP 10 (CŨ)
corr = df.drop('Time', axis=1).corr()['Class'].abs().sort_values(ascending=False)[1:11]
plt.figure(figsize=(8,10))
sns.heatmap(corr.to_frame().T, annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title('Hình 4: Top 10 đặc trưng tương quan với gian lận')
plt.tight_layout()
plt.savefig('reports/04_corr_top10.png', dpi=300, bbox_inches='tight')
plt.close()

# CHART 5: V14-V17 COMPARISON (MỚI - Quan trọng nhất)
v14_17 = ['V14','V15','V16','V17']
fig, axes = plt.subplots(2, 2, figsize=(12,10))
for i, feat in enumerate(v14_17):
    row, col = i//2, i%2
    sns.kdeplot(df[df['Class']==0][feat], shade=True, color='blue', ax=axes[row,col], alpha=0.5)
    sns.kdeplot(df[df['Class']==1][feat], shade=True, color='red', ax=axes[row,col], alpha=0.7)
    axes[row,col].set_title(f'{feat} Normal vs Fraud')
plt.suptitle('Hình 5: Phân bố V14-V17 (Top correlated features)', fontsize=14)
plt.tight_layout()
plt.savefig('reports/05_v14_v17_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# CHART 6: AMOUNT PER HOUR FRAUD (MỚI)
fig, ax = plt.subplots(figsize=(12,6))
fraud_data = df[df['Class']==1]
scatter = ax.scatter(fraud_data['hour'], fraud_data['Amount'], 
                    c=fraud_data['Amount'], cmap='Reds', alpha=0.7, s=30)
ax.set_title('Hình 6: Gian lận theo giờ và số tiền')
ax.set_xlabel('Giờ trong ngày')
ax.set_ylabel('Số tiền (USD)')
plt.colorbar(scatter, label='Số tiền')
plt.savefig('reports/06_fraud_amount_hour.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 6 CHARTS FULL HOÀN THÀNH:")
print("reports/ 01_fraud_pie.png")
print("reports/ 02_amount_hist.png") 
print("reports/ 03_time_hour.png")
print("reports/ 04_corr_top10.png")
print("reports/ 05_v14_v17_pca.png")
print("reports/ 06_fraud_amount_hour.png")