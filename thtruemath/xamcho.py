import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dữ liệu nhiệt độ trung bình của 12 tháng (giả định)
temperature_data = {
    "month": ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"],
    "temperature": [30, 28, 25, 22, 24, 27, 29, 31, 30, 26, 24, 2]  # Nhiệt độ trung bình
}

# Tạo DataFrame
df_temperature = pd.DataFrame(temperature_data)

# Vẽ biểu đồ cột biểu thị nhiệt độ trung bình
plt.figure(figsize=(10, 6))  # Kích thước hình vẽ
sns.barplot(x="month", y="temperature", data=df_temperature)

# Tùy chỉnh biểu đồ
plt.title("Thứ hai là ngày đầu tuần", fontsize=16)
plt.xlabel("Tháng", fontsize=14)
plt.ylabel("Nhiệt độ (°C)", fontsize=14)
plt.xticks(rotation=45)  # Xoay nhãn tháng để dễ đọc
plt.tight_layout()  # Đảm bảo bố cục không bị cắt

# Hiển thị biểu đồ
plt.show()


