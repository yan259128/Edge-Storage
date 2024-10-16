import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = "cpu_bandwidth_usage.csv"
df = pd.read_csv(file_path)

# 提取时间、自变量和因变量
time = df["Time (s)"]
cpu_usage = df["CPU (%)"]
netio = df["NetIO (Sent)"]


# 将带宽转换为数字（移除MB或kB等单位）
def convert_netio_to_mb(netio_str):
    sent, received = netio_str.split(" / ")

    # 处理发送流量
    if "MB" in sent:
        sent_value = float(sent.replace("MB", "").strip())
    elif "kB" in sent:
        sent_value = float(sent.replace("kB", "").strip()) / 1024
    else:
        sent_value = 0

    # 处理接收流量
    if "MB" in received:
        received_value = float(received.replace("MB", "").strip())
    elif "kB" in received:
        received_value = float(received.replace("kB", "").strip()) / 1024
    else:
        received_value = 0

    return sent_value, received_value


# 创建新的列，存放发送和接收流量
df["NetIO_Sent_MB"], df["NetIO_Received_MB"] = zip(*df["NetIO (Sent)"].apply(convert_netio_to_mb))

# 创建一个新的图表
plt.figure(figsize=(10, 6))

# 绘制 CPU 占用率折线图
plt.plot(time, cpu_usage, label="CPU Usage (%)", color="blue", marker="o")

# 绘制网络发送流量折线图
plt.plot(time, df["NetIO_Sent_MB"], label="NetIO Sent (MB)", color="green", marker="x")

# 绘制网络接收流量折线图
plt.plot(time, df["NetIO_Received_MB"], label="NetIO Received (MB)", color="red", marker="s")

# 设置图表标题和标签
plt.title("CPU Usage and Network I/O Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
