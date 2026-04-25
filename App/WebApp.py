from Analysis import AnalyzePicture, ShowMask, AnalizeMask, GetDictData
from Library import App, Temp, Shutil, OS, CV2, Torch, NP
from Caching import BuildModel
from Export import GetResults, GetImageResults
from Visualize import ShowVisuals
from db_integration import (
    db_integration, show_database_dashboard, show_analysis_history, 
    show_workorder_dashboard, show_settings_dashboard, show_pipeline_ledger,
    show_admin_dispatch_center, show_worker_task_center
)
from db_service import user_service, workorder_service, audit_service
from HealthScore import calc_health_score
from Explainability import build_single_explainability, build_ensemble_explainability, render_explainability_panel
import time
import multiprocessing as mp
from datetime import datetime

# ── 配置 ──────────────────────────────────────────────────────────
MODEL_PROFILES = {
    "YOLO11L-Seg.pt": "高速优先（轻量）",
    "YOLO9C-Seg.pt": "平衡模式（推荐）",
    "YOLO8L-Seg.pt": "精度优先（重型）",
}

DEFAULT_MODEL_ORDER = ["YOLO11L-Seg.pt", "YOLO9C-Seg.pt", "YOLO8L-Seg.pt"]
DEFAULT_NORMAL_MODEL = "YOLO9C-Seg.pt"
DEFAULT_FAST_MODEL = "YOLO11L-Seg.pt"

# ── 核心 UI 样式 ──────────────────────────────────────────────────
def load_css():
    App.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { border-radius: 8px; font-weight: 500; }
        .task-card { 
            background: white; padding: 1.2rem; border-radius: 12px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 1rem;
        }
        .status-badge {
            padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold;
        }
        h1, h2, h3 { color: #1e293b; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f5f9; border-radius: 8px 8px 0 0; padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    if "logged_in" not in App.session_state:
        App.session_state["logged_in"] = False
    if "user_info" not in App.session_state:
        App.session_state["user_info"] = None
    if "active_tab" not in App.session_state:
        App.session_state["active_tab"] = "调度中心"
    if "WorkDir" not in App.session_state:
        work = OS.path.join(Temp.gettempdir(), "StreamlitSewer")
        if OS.path.exists(work): Shutil.rmtree(work)
        OS.makedirs(work)
        App.session_state["WorkDir"] = work
        App.session_state["MODELS"] = {}
        App.session_state["MODELS_LOADED"] = False
    # 结果缓存
    for key in ["VIDEO_RESULTS", "IMAGE_RESULTS", "IMAGE_HS", "VIDEO_HS"]:
        if key not in App.session_state: App.session_state[key] = None
    if "INSPECTION_IMAGE_RESULTS" not in App.session_state:
        App.session_state["INSPECTION_IMAGE_RESULTS"] = []
    if "REPAIRED_RESULT_KEYS" not in App.session_state:
        App.session_state["REPAIRED_RESULT_KEYS"] = []

# ── 模型加载 ──────────────────────────────────────────────────────
def preload_models():
    if App.session_state.get("MODELS_LOADED"):
        return App.session_state["MODELS"]
    
    models_dir = OS.path.join(OS.path.dirname(OS.path.dirname(OS.path.abspath(__file__))), "Models")
    loaded = {}
    with App.spinner("🤖 正在启动智能检测引擎..."):
        for name in DEFAULT_MODEL_ORDER:
            path = OS.path.join(models_dir, name)
            if OS.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                    model = BuildModel(content, ".pt", App.session_state["WorkDir"], "auto")
                    if model: loaded[name] = model
                except: pass
    App.session_state["MODELS"] = loaded
    App.session_state["MODELS_LOADED"] = True
    return loaded

# ── 业务逻辑封装 ──────────────────────────────────────────────────
def get_mask_from_result(result, target_hw):
    target_h, target_w = target_hw
    if result.masks is None: return NP.zeros((target_h, target_w), dtype=NP.uint8)
    mask = (result.masks.data[0].cpu().numpy() * 255).astype(NP.uint8)
    if mask.shape != (target_h, target_w):
        mask = CV2.resize(mask, (target_w, target_h), interpolation=CV2.INTER_NEAREST)
    return mask

def run_smart_inference(image_input, mode="平衡模式"):
    models = App.session_state.get("MODELS", {})
    if not models:
        models = preload_models()
    if not models:
        return None
    
    # 映射档位到实际逻辑
    if mode == "深度精检":
        # 集成模式
        image_bgr = CV2.imread(image_input) if isinstance(image_input, str) else image_input.copy()
        h, w = image_bgr.shape[:2]
        masks = []
        start = time.time()
        for m in models.values():
            res = m(image_bgr, conf=0.25)[0]
            masks.append(get_mask_from_result(res, (h, w)))
        
        fused = (NP.sum([(m > 127).astype(NP.uint8) for m in masks], axis=0) >= 2).astype(NP.uint8) * 255
        analysis = AnalizeMask(fused)
        if analysis:
            mask_img, ellipse, shape, ar, orient, deform = analysis
            draw_mask = ShowMask(mask_img, ellipse, shape)
            data = GetDictData(shape, ar, orient, deform)
        else:
            draw_mask, data = fused, GetDictData(None, None, None, None)
        
        return {
            "plot": CV2.cvtColor(CV2.addWeighted(image_bgr, 1.0, CV2.merge([NP.zeros_like(fused), fused, NP.zeros_like(fused)]), 0.45, 0), CV2.COLOR_BGR2RGB),
            "mask": draw_mask, "data": data, "elapsed": round((time.time()-start)*1000, 1),
            "risk": data.get("RiskLevel")[0] if data.get("RiskLevel") else "低"
        }
    else:
        # 单模型模式
        m_name = "YOLO9C-Seg.pt" if mode == "平衡模式" else "YOLO11L-Seg.pt"
        if m_name not in models: m_name = list(models.keys())[0]
        model = models[m_name]
        
        start = time.time()
        res = model(image_input, conf=0.25)[0]
        (plot, mask, ellipse), data = AnalyzePicture(res)
        return {
            "plot": plot, "mask": ShowMask(mask, ellipse, data.get("Shape")[0]), 
            "data": data, "elapsed": round((time.time()-start)*1000, 1),
            "risk": data.get("RiskLevel")[0] if data.get("RiskLevel") else "低"
        }

# ── 页面组件 ──────────────────────────────────────────────────────
def _collect_files_from_path(folder_path: str):
    """递归收集指定目录下所有支持的图片/视频文件路径"""
    supported = {".jpg", ".jpeg", ".png", ".mp4", ".avi"}
    collected = []
    if OS.path.isfile(folder_path):
        if OS.path.splitext(folder_path)[1].lower() in supported:
            collected.append(folder_path)
    elif OS.path.isdir(folder_path):
        for root, _, fnames in OS.walk(folder_path):
            for fn in fnames:
                if OS.path.splitext(fn)[1].lower() in supported:
                    collected.append(OS.path.join(root, fn))
    return collected

def _result_action_suggestion(data):
    suggestion = (data or {}).get("ActionSuggestion")
    if isinstance(suggestion, list):
        return suggestion[0] if suggestion else "请尽快安排人工复核。"
    if isinstance(suggestion, str):
        return suggestion
    return "请尽快安排人工复核。"

def _create_repair_workorder(file_name, risk, data):
    from database import PipelineSegment
    session = workorder_service.db_manager.get_session()
    try:
        seg = session.query(PipelineSegment).first()  # 默认取首个管段（演示）
        workorder_service.create_work_order(
            title=f"巡检报修: {file_name}",
            description=f"AI 自动检出风险：{risk}。诊断结论：{_result_action_suggestion(data)}",
            risk_level=risk,
            segment_id=seg.id if seg else None,
        )
        audit_service.log_action(
            App.session_state["user_info"]["id"],
            App.session_state["user_info"]["username"],
            "报修",
            "巡检",
            f"为 {file_name} 创建了报修单",
        )
        return True
    finally:
        workorder_service.db_manager.close_session(session)

def _result_key(item):
    rid = item.get("id")
    if rid:
        return str(rid)
    return f"{item.get('name','')}-{item.get('source','')}"

def _render_inspection_results():
    results = App.session_state.get("INSPECTION_IMAGE_RESULTS", [])
    if not results:
        return

    App.markdown("#### 🧾 本次巡检结果")
    repaired_keys = set(App.session_state.get("REPAIRED_RESULT_KEYS", []))

    high_candidates = [it for it in results if it["res"].get("risk") in ["高", "中"] and _result_key(it) not in repaired_keys]
    if high_candidates:
        if App.button(f"🧰 高风险一键批量报修（{len(high_candidates)}条）", key="repair_batch_high", type="primary", use_container_width=True):
            created = 0
            try:
                for item in high_candidates:
                    res = item["res"]
                    if _create_repair_workorder(item["name"], res.get("risk", "中"), res.get("data") or {}):
                        repaired_keys.add(_result_key(item))
                        created += 1
                App.session_state["REPAIRED_RESULT_KEYS"] = list(repaired_keys)
                App.success(f"✅ 已批量生成 {created} 条报修工单")
                App.session_state["active_tab"] = "调度中心"
                App.rerun()
            except Exception as e:
                App.error(f"批量报修失败：{e}")

    for idx, item in enumerate(results):
        res = item["res"]
        with App.container(border=True):
            c1, c2, c3 = App.columns([2, 2, 2])
            c1.image(res["plot"], caption=f"诊断结果: {item['name']}")
            c2.dataframe(res["data"], hide_index=True)
            with c3:
                App.markdown(f"**风险等级: {res['risk']}**")
                row_key = _result_key(item)
                if row_key in repaired_keys:
                    App.success("已生成工单")
                elif res["risk"] in ["高", "中"]:
                    if App.button("🛠️ 立即报修", key=f"repair_cached_{idx}_{row_key}", use_container_width=True):
                        try:
                            if _create_repair_workorder(item["name"], res["risk"], res["data"]):
                                repaired_keys.add(row_key)
                                App.session_state["REPAIRED_RESULT_KEYS"] = list(repaired_keys)
                                App.success("✅ 已自动生成维修工单！")
                                App.session_state["active_tab"] = "调度中心"
                                App.rerun()
                        except Exception as e:
                            App.error(f"创建工单失败：{e}")

def render_inspection_page():
    App.markdown("### 🔍 巡检上传与诊断")
    image_results = []

    upload_tab, path_tab = App.tabs(["📤 文件上传", "📁 路径批量导入"])

    # ── 公共设置 ──────────────────────────────────────────────────
    inspect_mode = App.selectbox(
        "检测精度", ["快速扫描", "平衡模式", "深度精检"], index=1,
        help="深度精检将调用多个模型，耗时较长但最准确"
    )

    # ── Tab 1: 原有文件上传 ───────────────────────────────────────
    with upload_tab:
        files = App.file_uploader("上传现场照片或视频", ["JPG", "PNG", "MP4", "AVI"], accept_multiple_files=True)
        run_upload = files and App.button("🚀 开始自动化巡检诊断", type="primary", use_container_width=True, key="btn_upload")

    # ── Tab 2: 路径批量导入 ───────────────────────────────────────
    with path_tab:
        App.caption("输入本地文件夹路径或单个文件路径，支持递归扫描子目录")
        path_input = App.text_area(
            "文件/文件夹路径（每行一个）",
            placeholder="例如：\nD:\\inspection\\2024-04-25\nD:\\inspection\\single.jpg",
            height=120,
            key="batch_path_input"
        )
        col_scan, col_run = App.columns([1, 1])
        with col_scan:
            do_scan = App.button("🔎 扫描文件", use_container_width=True, key="btn_scan")
        with col_run:
            run_path = App.button("🚀 开始批量诊断", type="primary", use_container_width=True, key="btn_path_run")

        if do_scan and path_input.strip():
            scanned = []
            for line in path_input.strip().splitlines():
                line = line.strip()
                if line:
                    found = _collect_files_from_path(line)
                    scanned.extend(found)
            App.session_state["_batch_scanned_paths"] = scanned
            if scanned:
                App.success(f"共扫描到 {len(scanned)} 个文件")
                App.dataframe(
                    [{"文件路径": p, "类型": "视频" if p.lower().endswith((".mp4", ".avi")) else "图片"} for p in scanned],
                    use_container_width=True, hide_index=True
                )
            else:
                App.warning("未找到支持的文件，请检查路径是否正确")

    # ── 处理逻辑（上传模式）──────────────────────────────────────
    if run_upload and files:
        image_results = []
        for f in files:
            is_video = f.name.lower().endswith(('.mp4', '.avi'))
            save_path = OS.path.join(App.session_state["WorkDir"], f.name)
            with open(save_path, "wb") as out: out.write(f.read())
            
            if is_video:
                App.info(f"正在处理视频: {f.name}...")
                # 简化视频处理：每隔10帧抽稀
                cap = CV2.VideoCapture(save_path)
                v_res = []
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    if idx % 10 == 0:
                        res = run_smart_inference(frame, inspect_mode)
                        v_res.append(res)
                    idx += 1
                cap.release()
                App.success(f"视频 {f.name} 处理完成，检出 {len(v_res)} 个关键点")
                # 存储到历史
                db_integration.save_video_analysis(save_path, [{"Plot": r["plot"], "Mask": r["mask"], "Shape": r["data"]["Shape"][0], "RiskLevel": r["risk"]} for r in v_res], {"mode": inspect_mode})
            else:
                res = run_smart_inference(save_path, inspect_mode)
                if isinstance(res, dict):
                    image_results.append({"id": f"upload_{len(image_results)}", "name": f.name, "res": res, "source": "upload"})

    # ── 处理逻辑（路径批量模式）──────────────────────────────────
    if run_path:
        batch_paths = App.session_state.get("_batch_scanned_paths")
        if not batch_paths:
            # 未扫描则直接从输入框解析
            if path_input.strip():
                batch_paths = []
                for line in path_input.strip().splitlines():
                    line = line.strip()
                    if line:
                        batch_paths.extend(_collect_files_from_path(line))
        if not batch_paths:
            App.warning("请先输入路径并点击【扫描文件】，或直接输入路径后点击批量诊断")
        else:
            App.info(f"开始批量诊断，共 {len(batch_paths)} 个文件...")
            progress = App.progress(0, text="准备中...")
            total = len(batch_paths)
            path_image_results = []
            for i, fpath in enumerate(batch_paths):
                fname = OS.path.basename(fpath)
                progress.progress((i + 1) / total, text=f"正在处理 ({i+1}/{total}): {fname}")
                is_video = fpath.lower().endswith((".mp4", ".avi"))
                if is_video:
                    cap = CV2.VideoCapture(fpath)
                    v_res = []
                    idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        if idx % 10 == 0:
                            res = run_smart_inference(frame, inspect_mode)
                            v_res.append(res)
                        idx += 1
                    cap.release()
                    db_integration.save_video_analysis(fpath, [{"Plot": r["plot"], "Mask": r["mask"], "Shape": r["data"]["Shape"][0], "RiskLevel": r["risk"]} for r in v_res], {"mode": inspect_mode})
                    App.success(f"✅ 视频 {fname} 处理完成，检出 {len(v_res)} 个关键点")
                else:
                    res = run_smart_inference(fpath, inspect_mode)
                    if isinstance(res, dict):
                        path_image_results.append({"id": f"path_{i}", "name": fname, "res": res, "source": "path"})
            progress.empty()
            App.success(f"🎉 批量诊断完成！共处理 {total} 个文件")
            App.session_state.pop("_batch_scanned_paths", None)
            App.session_state["INSPECTION_IMAGE_RESULTS"] = path_image_results
            App.session_state["REPAIRED_RESULT_KEYS"] = []

    if image_results:
        App.session_state["INSPECTION_IMAGE_RESULTS"] = image_results
        App.session_state["REPAIRED_RESULT_KEYS"] = []
    _render_inspection_results()

def login_page():
    App.markdown("<div style='text-align:center;margin-top:100px;'><h1>🏙️ 城市下水道智能运维平台</h1><p>请登录以访问您的工作台</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = App.columns([1, 2, 1])
    with col2:
        with App.form("login"):
            u = App.text_input("账号")
            p = App.text_input("密码", type="password")
            if App.form_submit_button("登录系统", use_container_width=True):
                user = user_service.verify_user(u, p)
                if user:
                    App.session_state.update({"logged_in": True, "user_info": {"id": user.id, "username": user.username, "role": user.role}})
                    App.rerun()
                else: App.error("账号或密码错误")
    App.info("💡 提示：管理员账号 admin / 123456，工人账号 worker / 123456")

# ── 主程序入口 ────────────────────────────────────────────────────
init_session_state()
load_css()

if not App.session_state["logged_in"]:
    login_page()
    App.stop()

user = App.session_state["user_info"]
with App.sidebar:
    App.markdown(f"### 👤 {user['username']}")
    App.caption(f"身份: {'系统管理员' if user['role'] == 'admin' else '现场工程师'}")
    if App.button("退出登录", use_container_width=True):
        App.session_state["logged_in"] = False
        App.rerun()
    App.markdown("---")
    App.markdown("#### 📅 今日概况")
    App.info(f"日期: {datetime.now().strftime('%Y-%m-%d')}")

def render_worker_acceptance_page():
    """工人端现场验收页面"""
    App.markdown("### 📸 现场维修验收")
    
    active_wo = App.session_state.get("active_wo_for_detect")
    if not active_wo:
        App.info("💡 请先在【任务中心】选择一个待验收的工单，点击【拍照验收】按钮。")
        return

    App.success(f"正在验收工单: **{active_wo['code']}** (管段: {active_wo['segment']})")
    
    up_col, set_col = App.columns([3, 1])
    with set_col:
        mode = App.selectbox("验证精度", ["快速扫描", "平衡模式", "深度精检"], index=1)
    with up_col:
        f = App.file_uploader("拍摄或上传维修后照片", ["JPG", "PNG"])

    if f and App.button("🔬 开始 AI 验收分析", type="primary", use_container_width=True):
        save_path = OS.path.join(App.session_state["WorkDir"], f"verify_{f.name}")
        with open(save_path, "wb") as out: out.write(f.read())
        
        res = run_smart_inference(save_path, mode)
        if not isinstance(res, dict) or res.get("plot") is None:
            App.error("AI 验收分析失败：模型未就绪或图像未识别成功，请稍后重试。")
            return
        data = res.get("data") or {}
        deform_list = data.get("Deformation") or [None]
        risk_level = res.get("risk", "未知")
        
        with App.container(border=True):
            c1, c2 = App.columns([1, 1])
            c1.image(res["plot"], caption="现场维修后状态")
            with c2:
                # 获取健康评分
                from HealthScore import calc_health_score
                # 构造简易结果列表供评分
                mock_res = [{"Deformation": deform_list[0], "RiskLevel": risk_level}]
                hs = calc_health_score(mock_res, 0.3, 0.15)
                
                App.metric("管道健康得分", f"{hs['score']}/100")
                App.markdown(f"**诊断结论:** {hs['level']}")
                
                if hs["score"] >= 80:
                    App.success("🎉 AI 评估合格！维修效果良好。")
                    if App.button("✅ 提交验收并闭环工单", type="primary", use_container_width=True):
                        workorder_service.update_work_order(
                            active_wo["id"], status="已修复", 
                            processing_notes=f"现场验收通过。AI 得分: {hs['score']} ({hs['level']})。"
                        )
                        audit_service.log_action(App.session_state["user_info"]["id"], App.session_state["user_info"]["username"], 
                                               "验收提交", "现场验收", f"工单 {active_wo['code']} 验收合格并提交")
                        del App.session_state["active_wo_for_detect"]
                        App.success("工单已提交，正在返回任务中心...")
                        time.sleep(1.5)
                        App.rerun()
                else:
                    App.error("⚠️ AI 评估得分较低，建议检查是否修复彻底。")
                    if App.button("强制提交说明", use_container_width=True):
                        App.warning("请联系管理员进行人工审核。")

# 根据角色定义 Tab
if user["role"] == "admin":
    ADMIN_TABS = ["🛰️ 调度中心", "🔍 巡检检测", "📂 历史档案", "🧰 工单管理", "🗂️ 管网台账", "⚙️ 系统管理"]
    # 支持通过 session_state 程序化切换（如调度中心"处理"按钮跳转）
    if "admin_tab_radio" not in App.session_state:
        App.session_state["admin_tab_radio"] = ADMIN_TABS[0]
    if "admin_active_tab" not in App.session_state:
        App.session_state["admin_active_tab"] = App.session_state["admin_tab_radio"]
    # 若外部请求跳转到工单管理，在 radio 实例化前写入目标值
    _jump = App.session_state.pop("active_tab", None)
    if _jump == "工单管理":
        App.session_state["admin_tab_radio"] = "🧰 工单管理"
    elif _jump == "调度中心":
        App.session_state["admin_tab_radio"] = "🛰️ 调度中心"

    selected_tab = App.radio(
        "导航", ADMIN_TABS,
        horizontal=True,
        label_visibility="collapsed",
        key="admin_tab_radio",
    )
    App.session_state["admin_active_tab"] = selected_tab
    App.markdown("---")

    if selected_tab == "🛰️ 调度中心":   show_admin_dispatch_center()
    elif selected_tab == "🔍 巡检检测": render_inspection_page()
    elif selected_tab == "📂 历史档案": show_analysis_history()
    elif selected_tab == "🧰 工单管理": show_workorder_dashboard()
    elif selected_tab == "🗂️ 管网台账": show_pipeline_ledger()
    elif selected_tab == "⚙️ 系统管理": show_settings_dashboard()
else:
    tabs = App.tabs(["📋 任务中心", "📸 现场验收", "🗺️ 管网查询"])
    with tabs[0]: show_worker_task_center(user["id"])
    with tabs[1]: render_worker_acceptance_page()
    with tabs[2]: show_pipeline_ledger()
