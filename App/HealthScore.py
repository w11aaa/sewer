"""
管道健康度评分模块
基于变形率、风险等级、帧占比计算 0-100 综合健康评分
"""


def calc_health_score(results, risk_threshold_high=0.3, risk_threshold_mid=0.15):
    """
    计算管道健康度评分。

    参数
    ----
    results : list[dict]
        每帧/每张图的检测结果，需含 RiskLevel 和 Deformation 字段。
    risk_threshold_high : float
        高风险变形率阈值（来自系统配置）。
    risk_threshold_mid : float
        中风险变形率阈值（来自系统配置）。

    返回
    ----
    dict 包含:
        score        : int   0-100
        level        : str   "优良" / "警告" / "危险"
        color        : str   "green" / "orange" / "red"
        high_ratio   : float 高风险帧占比 0-1
        mid_ratio    : float 中风险帧占比 0-1
        low_ratio    : float 低风险帧占比 0-1
        avg_deform   : float 平均变形率
        max_deform   : float 最大变形率
        detail       : str   评分说明
    """
    if not results:
        return _empty_score()

    total = len(results)
    high_cnt = mid_cnt = low_cnt = 0
    deform_vals = []

    for item in results:
        risk = (item.get("RiskLevel") or "").strip()
        if risk == "高":
            high_cnt += 1
        elif risk == "中":
            mid_cnt += 1
        else:
            low_cnt += 1

        d = item.get("Deformation")
        try:
            d = float(d)
            if 0.0 <= d <= 1.0:
                deform_vals.append(d)
        except (TypeError, ValueError):
            pass

    high_ratio = high_cnt / total
    mid_ratio  = mid_cnt  / total
    low_ratio  = low_cnt  / total
    avg_deform = sum(deform_vals) / len(deform_vals) if deform_vals else 0.0
    max_deform = max(deform_vals) if deform_vals else 0.0

    # ── 扣分规则 ──────────────────────────────────────────────
    # 高风险帧占比：每 1% 扣 1.5 分，上限扣 60 分
    penalty_high = min(high_ratio * 100 * 1.5, 60.0)
    # 中风险帧占比：每 1% 扣 0.5 分，上限扣 25 分
    penalty_mid  = min(mid_ratio  * 100 * 0.5, 25.0)
    # 平均变形率超过中风险阈值的部分额外扣分
    deform_excess = max(avg_deform - risk_threshold_mid, 0.0)
    penalty_deform = min(deform_excess / (risk_threshold_high - risk_threshold_mid + 1e-6) * 15.0, 15.0)

    raw_score = 100.0 - penalty_high - penalty_mid - penalty_deform
    score = max(0, min(100, round(raw_score)))

    if score >= 75:
        level, color = "优良", "green"
    elif score >= 45:
        level, color = "警告", "orange"
    else:
        level, color = "危险", "red"

    detail = (
        f"高风险帧 {high_ratio*100:.1f}%（扣 {penalty_high:.1f} 分）｜"
        f"中风险帧 {mid_ratio*100:.1f}%（扣 {penalty_mid:.1f} 分）｜"
        f"变形率惩罚（扣 {penalty_deform:.1f} 分）"
    )

    return {
        "score":      score,
        "level":      level,
        "color":      color,
        "high_ratio": high_ratio,
        "mid_ratio":  mid_ratio,
        "low_ratio":  low_ratio,
        "avg_deform": avg_deform,
        "max_deform": max_deform,
        "detail":     detail,
    }


def _empty_score():
    return {
        "score": 100, "level": "优良", "color": "green",
        "high_ratio": 0.0, "mid_ratio": 0.0, "low_ratio": 0.0,
        "avg_deform": 0.0, "max_deform": 0.0, "detail": "无检测数据",
    }
