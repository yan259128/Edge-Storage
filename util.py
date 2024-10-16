# 计算交易期望
def calculate_transaction_expectations(inclination,preference, satisfaction):
    """
    输入：
        意愿系数：inclination
        偏好度：preference
        满意度：satisfaction
    输出：
        交易期望：transaction_expectations
    """
    return ((preference+satisfaction)/2)*inclination

