"""
数据库集成模块
将数据库功能集成到Streamlit应用中
"""

import streamlit as App
from database import db_ops
from db_service import (
    detection_service, model_service, statistics_service, 
    user_service, audit_service, workorder_service, segment_service
)
from Export import GetWorkOrderPDF, GetWorkOrderExcel
from datetime import datetime
import json
import os

class DatabaseIntegration:
    """数据库集成类"""
    
    def __init__(self):
        self._initialized = False
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        if self._initialized:
            return
            
        try:
            db_ops.init_database()
            self._initialized = True
            # 只在第一次初始化时显示成功消息
            if not hasattr(App.session_state, 'db_init_message_shown'):
                App.success("✅ 数据库初始化成功")
                App.session_state['db_init_message_shown'] = True
        except Exception as e:
            App.error(f"❌ 数据库初始化失败: {str(e)}")
            # 记录错误到session state，避免重复显示
            App.session_state['db_init_error'] = str(e)
    
    def save_image_analysis(self, image_path, analysis_result, plot_image, mask_image):
        """保存图像分析结果"""
        try:
            # 创建任务
            task_id = detection_service.create_task(
                task_name=f"图像分析_{os.path.basename(image_path)}",
                task_type="image",
                input_file_path=image_path,
                user_id=1,  # 默认用户ID
                model_id=1,  # 默认模型ID
                parameters={"analysis_type": "single_image"}
            )
            
            # 更新任务状态
            detection_service.update_task_status(task_id, "processing")
            
            # 保存结果
            images_data = {
                "original": None,  # 原始图像路径已知，不存储
                "segmented": plot_image,
                "mask": mask_image
            }
            
            result_id = detection_service.save_detection_result(
                task_id=task_id,
                frame_number=1,
                analysis_data=analysis_result,
                images_data=images_data
            )
            
            # 完成任务
            detection_service.update_task_status(task_id, "completed", 100.0)
            
            return task_id, result_id
            
        except Exception as e:
            App.error(f"保存图像分析结果失败: {str(e)}")
            return None, None
    
    def save_video_analysis(self, video_path, video_results, processing_parameters):
        """保存视频分析结果"""
        try:
            # 创建任务
            task_id = detection_service.create_task(
                task_name=f"视频分析_{os.path.basename(video_path)}",
                task_type="video",
                input_file_path=video_path,
                user_id=1,  # 默认用户ID
                model_id=1,  # 默认模型ID
                parameters=processing_parameters
            )
            
            # 更新任务状态
            detection_service.update_task_status(task_id, "processing")
            
            # 保存所有帧的结果
            for i, result_data in enumerate(video_results):
                images_data = {
                    "segmented": result_data.get("Plot"),
                    "mask": result_data.get("Mask"),
                    "output": result_data.get("Draw")
                }
                
                # 构建分析数据
                analysis_data = {
                    "Shape": [result_data.get("Shape")],
                    "AspectRatio": [result_data.get("AspectRatio")],
                    "Orientation": [result_data.get("Orientation")],
                    "Deformation": [result_data.get("Deformation")],
                    "Conclusion": ["Normal" if result_data.get("Shape") == "Circle" else "Deformed"],
                    "RiskLevel": [result_data.get("RiskLevel")],
                    "ActionSuggestion": [result_data.get("ActionSuggestion")]
                }
                
                detection_service.save_detection_result(
                    task_id=task_id,
                    frame_number=i + 1,
                    analysis_data=analysis_data,
                    images_data=images_data
                )
                
                # 更新进度
                progress = ((i + 1) / len(video_results)) * 100
                detection_service.update_task_status(task_id, "processing", progress)
            
            # 完成任务
            detection_service.update_task_status(task_id, "completed", 100.0)
            
            return task_id
            
        except Exception as e:
            App.error(f"保存视频分析结果失败: {str(e)}")
            return None
    
    def get_analysis_history(self, limit=10):
        """获取分析历史"""
        try:
            session = detection_service.db_manager.get_session()
            try:
                from database import DetectionTask, DetectionResult
                
                tasks = session.query(DetectionTask).order_by(
                    DetectionTask.created_at.desc()
                ).limit(limit).all()
                
                history = []
                for task in tasks:
                    # 获取任务结果统计
                    result_count = session.query(DetectionResult).filter(
                        DetectionResult.task_id == task.id
                    ).count()
                    
                    history.append({
                        'id': task.id,
                        'name': task.task_name,
                        'type': task.task_type,
                        'status': task.status,
                        'progress': task.progress,
                        'result_count': result_count,
                        'created_at': task.created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(task.created_at, 'strftime') else str(task.created_at),
                        'completed_at': task.completed_at.strftime('%Y-%m-%d %H:%M:%S') if task.completed_at and hasattr(task.completed_at, 'strftime') else (str(task.completed_at) if task.completed_at else None)
                    })
                
                return history
                
            finally:
                detection_service.db_manager.close_session(session)
                
        except Exception as e:
            App.error(f"获取分析历史失败: {str(e)}")
            return []
    
    def get_task_details(self, task_id):
        """获取任务详情"""
        try:
            session = detection_service.db_manager.get_session()
            try:
                from database import DetectionTask, DetectionResult
                
                task = session.query(DetectionTask).filter(DetectionTask.id == task_id).first()
                if not task:
                    return None
                
                results = session.query(DetectionResult).filter(
                    DetectionResult.task_id == task_id
                ).order_by(DetectionResult.frame_number).all()
                
                return {
                    'task': task,
                    'results': results
                }
                
            finally:
                detection_service.db_manager.close_session(session)
                
        except Exception as e:
            App.error(f"获取任务详情失败: {str(e)}")
            return None
    
    def get_statistics_dashboard(self):
        """获取统计仪表板数据"""
        try:
            # 获取任务统计
            task_stats = statistics_service.get_task_statistics(30)
            
            # 获取性能指标
            performance_metrics = statistics_service.get_performance_metrics(7)
            
            return {
                'task_statistics': task_stats,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            App.error(f"获取统计数据失败: {str(e)}")
            return None
    
    def export_task_data(self, task_id, export_format='json'):
        """导出任务数据"""
        try:
            if export_format == 'pdf':
                return self.export_task_pdf(task_id)
                
            task_details = self.get_task_details(task_id)
            if not task_details:
                return None
            
            task = task_details['task']
            results = task_details['results']
            
            if export_format == 'json':
                export_data = {
                    'task_info': {
                        'id': task.id,
                        'name': task.task_name,
                        'type': task.task_type,
                        'status': task.status,
                        'created_at': task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at),
                        'completed_at': task.completed_at.isoformat() if task.completed_at and hasattr(task.completed_at, 'isoformat') else (str(task.completed_at) if task.completed_at else None),
                        'parameters': json.loads(task.parameters) if task.parameters else None
                    },
                    'results': []
                }
                
            for result in results:
                extra_result_data = {}
                try:
                    extra_result_data = json.loads(result.result_data) if result.result_data else {}
                except Exception:
                    extra_result_data = {}

                risk_level = extra_result_data.get("RiskLevel")
                action_suggestion = extra_result_data.get("ActionSuggestion")
                if isinstance(risk_level, list):
                    risk_level = risk_level[0] if risk_level else None
                if isinstance(action_suggestion, list):
                    action_suggestion = action_suggestion[0] if action_suggestion else None

                result_data = {
                    'frame_number': result.frame_number,
                    'shape': result.shape,
                    'aspect_ratio': result.aspect_ratio,
                    'orientation': result.orientation,
                    'deformation': result.deformation,
                    'conclusion': result.conclusion,
                    'risk_level': risk_level,
                    'action_suggestion': action_suggestion,
                    'created_at': result.created_at.isoformat() if hasattr(result.created_at, 'isoformat') else str(result.created_at)
                }
                export_data['results'].append(result_data)
                
                return json.dumps(export_data, ensure_ascii=False, indent=2)
            
            return None
            
        except Exception as e:
            App.error(f"导出任务数据失败: {str(e)}")
            return None

    def export_task_pdf(self, task_id):
        """导出任务PDF报告"""
        try:
            from Export import GetPDF
            import numpy as np
            from PIL import Image
            from io import BytesIO
            
            task_details = self.get_task_details(task_id)
            if not task_details:
                return None
            
            results = task_details['results']
            data_for_pdf = []
            
            for result in results:
                extra_result_data = {}
                try:
                    extra_result_data = json.loads(result.result_data) if result.result_data else {}
                except Exception:
                    extra_result_data = {}

                risk_level = extra_result_data.get("RiskLevel")
                action_suggestion = extra_result_data.get("ActionSuggestion")
                if isinstance(risk_level, list):
                    risk_level = risk_level[0] if risk_level else None
                if isinstance(action_suggestion, list):
                    action_suggestion = action_suggestion[0] if action_suggestion else None

                # Get images
                images = detection_service.get_result_images(result.id)
                plot_img = None
                mask_img = None
                
                for img in images:
                    if img.image_type == 'segmented': 
                        pil_img = Image.open(BytesIO(img.image_data))
                        plot_img = np.array(pil_img)
                    elif img.image_type == 'output': 
                        pil_img = Image.open(BytesIO(img.image_data))
                        mask_img = np.array(pil_img)
                    elif img.image_type == 'mask' and mask_img is None:
                        pil_img = Image.open(BytesIO(img.image_data))
                        mask_img = np.array(pil_img)

                # If missing images, create black ones
                if plot_img is None: plot_img = np.zeros((100, 100, 3), dtype=np.uint8)
                if mask_img is None: mask_img = np.zeros((100, 100, 3), dtype=np.uint8)

                item = {
                    "Plot": plot_img,
                    "Draw": mask_img,
                    "Shape": result.shape,
                    "AspectRatio": result.aspect_ratio,
                    "Orientation": result.orientation,
                    "Deformation": result.deformation,
                    "RiskLevel": risk_level,
                    "ActionSuggestion": action_suggestion,
                }
                data_for_pdf.append(item)
            
            if not data_for_pdf:
                return None
                
            return GetPDF(data_for_pdf)
            
        except Exception as e:
            App.error(f"导出PDF失败: {str(e)}")
            return None
    
    def cleanup_old_data(self, days=30):
        """清理指定天数前的历史数据"""
        try:
            session = detection_service.db_manager.get_session()
            try:
                from database import DetectionTask, DetectionResult, ResultImage
                from datetime import datetime, timedelta
                
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # 获取要删除的任务
                old_tasks = session.query(DetectionTask).filter(
                    DetectionTask.created_at < cutoff_date
                ).all()
                
                deleted_tasks = 0
                deleted_results = 0
                deleted_images = 0
                
                # 如果没有要删除的任务，直接返回
                if not old_tasks:
                    return {
                        'deleted_tasks': 0,
                        'deleted_results': 0,
                        'deleted_images': 0
                    }
                
                for task in old_tasks:
                    # 获取该任务的所有结果
                    results = session.query(DetectionResult).filter(
                        DetectionResult.task_id == task.id
                    ).all()
                    
                    # 删除每个结果的相关图像
                    for result in results:
                        result_images = session.query(ResultImage).filter(
                            ResultImage.result_id == result.id
                        ).all()
                        
                        for img in result_images:
                            session.delete(img)
                            deleted_images += 1
                        
                        # 删除结果
                        session.delete(result)
                        deleted_results += 1
                    
                    # 删除任务
                    session.delete(task)
                    deleted_tasks += 1
                
                # 提交所有更改
                session.commit()
                
                return {
                    'deleted_tasks': deleted_tasks,
                    'deleted_results': deleted_results,
                    'deleted_images': deleted_images
                }
                
            except Exception as e:
                # 如果出错，回滚事务
                session.rollback()
                raise e
            finally:
                detection_service.db_manager.close_session(session)
                
        except Exception as e:
            App.error(f"数据清理失败: {str(e)}")
            return None

# 全局数据库集成实例
db_integration = DatabaseIntegration()

@App.fragment
def show_database_dashboard():
    """显示数据库仪表板"""
    App.markdown("### 📊 数据库仪表板")
    
    # 检查数据库初始化状态
    if hasattr(App.session_state, 'db_init_error'):
        App.error(f"数据库连接错误: {App.session_state['db_init_error']}")
        App.info("请检查数据库文件是否存在或重新启动应用")
        return
    
    # 获取统计数据
    try:
        dashboard_data = db_integration.get_statistics_dashboard()
        if not dashboard_data:
            App.warning("⚠️ 暂无统计数据，请先进行一些分析任务")
            return
        
        task_stats = dashboard_data['task_statistics']
        performance_metrics = dashboard_data['performance_metrics']
        
        # 显示统计卡片
        col1, col2, col3, col4 = App.columns(4)
        
        with col1:
            App.metric(
                "总任务数",
                task_stats.get('total_tasks', 0),
                delta=f"成功率: {task_stats.get('success_rate', 0):.1f}%"
            )
        
        with col2:
            App.metric(
                "完成任务",
                task_stats.get('completed_tasks', 0),
                delta=f"失败: {task_stats.get('failed_tasks', 0)}"
            )
        
        with col3:
            App.metric(
                "图像任务",
                task_stats.get('image_tasks', 0),
                delta=f"视频任务: {task_stats.get('video_tasks', 0)}"
            )
        
        with col4:
            avg_time = performance_metrics.get('average_processing_time', 0)
            App.metric(
                "检测结果",
                task_stats.get('total_results', 0),
                delta=f"平均处理时间: {avg_time:.1f}s"
            )
        
            
    except Exception as e:
        App.error(f"获取统计数据时发生错误: {str(e)}")
        App.info("请检查数据库连接或联系管理员")

@App.fragment
def show_analysis_history():
    """历史档案"""
    App.markdown("### 📂 历史档案查询")
    
    # 检查数据库连接
    if hasattr(App.session_state, 'db_init_error'):
        App.error("数据库连接错误，无法显示历史记录")
        return
    
    try:
        # 获取历史记录
        history = db_integration.get_analysis_history(20)
        if not history:
            App.info("📝 暂无分析历史记录，请先进行一些分析任务")
            return
        
        # 显示历史表格
        App.dataframe(
            history,
            width="stretch",
            hide_index=True,
            column_config={
                "id": App.column_config.NumberColumn("任务ID"),
                "name": App.column_config.TextColumn("任务名称"),
                "type": App.column_config.TextColumn("类型"),
                "status": App.column_config.TextColumn("状态"),
                "progress": App.column_config.ProgressColumn("进度"),
                "result_count": App.column_config.NumberColumn("结果数"),
                "created_at": App.column_config.DatetimeColumn("创建时间"),
                "completed_at": App.column_config.DatetimeColumn("完成时间")
            }
        )
        
        # 任务详情查看
        if history:
            selected_task_id = App.selectbox(
                "选择任务查看详情",
                [task['id'] for task in history],
                format_func=lambda x: f"任务 {x} - {next((t['name'] for t in history if t['id'] == x), '未知')}"
            )
            
            if selected_task_id:
                try:
                    task_details = db_integration.get_task_details(selected_task_id)
                    if task_details:
                        with App.expander("📋 任务详情", expanded=False):
                            task = task_details['task']
                            results = task_details['results']
                            
                            # 基本信息
                            col1, col2 = App.columns(2)
                            with col1:
                                App.write(f"**任务名称**: {task.task_name}")
                                App.write(f"**任务类型**: {task.task_type}")
                                App.write(f"**状态**: {task.status}")
                            with col2:
                                App.write(f"**进度**: {task.progress}%")
                                App.write(f"**结果数量**: {len(results)}")
                                if task.completed_at:
                                    App.write(f"**完成时间**: {task.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # 参数信息
                            if task.parameters:
                                try:
                                    params = json.loads(task.parameters)
                                    App.write("**处理参数**:")
                                    App.json(params)
                                except:
                                    App.write(f"**处理参数**: {task.parameters}")
                            
                            # 导出功能
                            export_col1, export_col2, export_col3 = App.columns(3)
                            with export_col1:
                                if App.button("📥 导出JSON数据", key=f"export_json_{selected_task_id}"):
                                    export_data = db_integration.export_task_data(selected_task_id)
                                    if export_data:
                                        App.download_button(
                                            "下载JSON数据",
                                            export_data,
                                            f"task_{selected_task_id}_data.json",
                                            "application/json"
                                        )
                                    else:
                                        App.error("导出失败，请重试")
                            
                            with export_col2:
                                if App.button("📄 导出PDF报告", key=f"export_pdf_{selected_task_id}"):
                                    with App.spinner("正在生成PDF报告..."):
                                        pdf_buffer = db_integration.export_task_data(selected_task_id, export_format='pdf')
                                        if pdf_buffer:
                                            App.download_button(
                                                "下载PDF报告",
                                                pdf_buffer,
                                                f"task_{selected_task_id}_report.pdf",
                                                "application/pdf"
                                            )
                                        else:
                                            App.error("PDF生成失败或无数据")

                            with export_col3:
                                if App.button("🗑️ 删除任务", key=f"delete_{selected_task_id}"):
                                    App.warning("⚠️ 此操作将永久删除该任务及其所有相关数据")
                                    if App.button("确认删除", key=f"confirm_delete_{selected_task_id}"):
                                        # 这里可以添加删除功能
                                        App.info("删除功能开发中...")
                    
                    else:
                        App.warning("⚠️ 无法获取任务详情")
                        
                except Exception as e:
                    App.error(f"获取任务详情时发生错误: {str(e)}")
                    
    except Exception as e:
        App.error(f"获取历史记录时发生错误: {str(e)}")
        App.info("请检查数据库连接或联系管理员")


def show_workorder_dashboard(assignee_id=None):
    """显示工单闭环模块"""
    App.markdown("### 🧰 工单闭环")

    if hasattr(App.session_state, 'db_init_error'):
        App.error("数据库连接错误，无法加载工单模块")
        return

    try:
        stats = workorder_service.get_work_order_stats(assignee_id=assignee_id)
        status_distribution = stats.get("status_distribution", {})

        c1, c2, c3, c4 = App.columns(4)
        c1.metric("工单总数", stats.get("total", 0))
        c2.metric("已复检", stats.get("closed", 0), f"闭环率 {stats.get('closed_rate', 0):.1f}%")
        c3.metric("待确认", status_distribution.get("待确认", 0))
        c4.metric("处理中", status_distribution.get("处理中", 0))

        if not assignee_id:
            _workorder_create_section()
        _workorder_list_and_update(assignee_id=assignee_id)

    except Exception as e:
        App.error(f"加载工单模块失败: {str(e)}")


@App.fragment
def _workorder_create_section():
    App.markdown("#### ⚙️ 自动建单（高风险）")

    # 关联管段（可选）
    segments = segment_service.list_segments()
    seg_options = {0: "不关联管段"}
    seg_options.update({s.id: f"{s.segment_code} — {s.location or '无位置'}" for s in segments})
    selected_seg_id = App.selectbox(
        "关联管段（可选）",
        list(seg_options.keys()),
        format_func=lambda x: seg_options[x],
        key="create_wo_segment",
    )
    selected_seg_id = selected_seg_id if selected_seg_id != 0 else None

    history = db_integration.get_analysis_history(30)
    task_options = [item["id"] for item in history if item.get("result_count", 0) > 0]
    if task_options:
        selected_task = App.selectbox(
            "选择分析任务",
            task_options,
            format_func=lambda x: f"任务{x} - {next((t['name'] for t in history if t['id'] == x), '未知')}"
        )
        if App.button("从该任务生成高风险工单", key="create_workorders_from_task"):
            created_count = workorder_service.create_work_orders_from_task(
                selected_task, risk_level="高", segment_id=selected_seg_id
            )
            if created_count > 0:
                App.success(f"已创建 {created_count} 条高风险工单")
                # 更新管段的上次检测时间
                if selected_seg_id:
                    segment_service.update_last_inspected(selected_seg_id)
            else:
                App.info("没有可新建的高风险结果（可能已建单或无高风险）")
    else:
        App.info("暂无可用于建单的分析任务")


@App.fragment
def _workorder_list_and_update(assignee_id=None):
    App.markdown("#### 📋 工单列表")
    status_filter = App.selectbox("状态筛选", ["全部", "待确认", "已派单", "处理中", "已修复", "已复检"], key="workorder_status_filter")
    
    # 获取工单列表
    orders = workorder_service.list_work_orders(status=status_filter, limit=300)
    
    if assignee_id:
        # 如果是工人端，只显示分配给该工人的工单
        orders = [o for o in orders if o.assignee == App.session_state["user_info"]["username"]]
    if not orders:
        App.info("暂无工单记录")
        return

    order_rows = []
    status_colors = {
        "待确认": "gray",
        "已派单": "blue",
        "处理中": "orange",
        "已修复": "green",
        "已复检": "violet"
    }

    for order in orders:
        status_c = status_colors.get(order.status, "gray")
        order_rows.append({
            "选择": False,
            "id": order.id,
            "工单编号": order.order_code,
            "管段": order.segment.segment_code if order.segment else "",
            "标题": order.title,
            "风险": order.risk_level,
            "优先级": order.priority,
            "状态": f"[{order.status}]", # 在编辑器中暂时无法直接渲染 HTML，用括号标识
            "负责人": order.assignee or "未指派",
            "处理记录": order.processing_notes or "",
            "创建时间": order.created_at.strftime('%m-%d %H:%M') if order.created_at else "",
            "更新时间": order.updated_at.strftime('%m-%d %H:%M') if order.updated_at else "",
        })

    # 为 UI 添加一点色彩提示
    if assignee_id:
        App.info(f"💡 你好 {App.session_state['user_info']['username']}，以下是分配给你的维修任务。")
    
    edited = App.data_editor(
        order_rows,
        column_config={
            "选择": App.column_config.CheckboxColumn("选择", default=False),
            "id": None, # 隐藏 ID
            "工单编号": App.column_config.TextColumn("编号", width="small", disabled=True),
            "管段": App.column_config.TextColumn("位置", width="small", disabled=True),
            "标题": App.column_config.TextColumn("任务描述", width="medium", disabled=True),
            "风险": App.column_config.SelectboxColumn("风险", options=["高", "中", "低"], width="small", disabled=True),
            "优先级": App.column_config.SelectboxColumn("优先级", options=["紧急", "高", "普通", "低"], width="small", disabled=True),
            "状态": App.column_config.TextColumn("当前状态状态", width="small", disabled=True),
            "负责人": App.column_config.TextColumn("负责人", width="small", disabled=True),
            "处理记录": App.column_config.TextColumn("最新记录", width="medium", disabled=True),
            "创建时间": App.column_config.TextColumn("下发时间", width="small", disabled=True),
            "更新时间": None, # 隐藏更新时间以精简
        },
        hide_index=True,
        use_container_width=True,
        key="workorder_table_editor",
    )

    selected_ids = [row["id"] for row in edited if row.get("选择")]
    selected_rows = [row for row in edited if row.get("选择")]

    if selected_ids:
        App.markdown(f"**已选 {len(selected_ids)} 条工单**")
        
        # 管理员显示批量指派面板
        user_info = App.session_state.get("user_info", {})
        if user_info.get("role") == "admin":
            with App.expander("🚀 批量指派与更新", expanded=True):
                bc1, bc2, bc3 = App.columns([2, 2, 1])
                with bc1:
                    all_workers = user_service.list_users(role="user", status="active")
                    worker_names = ["(保持不变)"] + [w.username for w in all_workers]
                    bulk_assignee = App.selectbox("批量指派给", worker_names, key="bulk_wo_assignee")
                with bc2:
                    bulk_status_options = ["(保持不变)", "待确认", "已派单", "处理中", "已修复", "已复检"]
                    bulk_next_status = App.selectbox("批量更新状态", bulk_status_options, key="bulk_wo_status")
                with bc3:
                    App.markdown("&nbsp;", unsafe_allow_html=True)
                    if App.button("⚡ 执行批量更新", key="bulk_wo_update_btn", type="primary", use_container_width=True):
                        success_count = 0
                        for wid in selected_ids:
                            update_params = {}
                            log_msg = []
                            
                            if bulk_assignee != "(保持不变)":
                                update_params["assignee"] = bulk_assignee
                                log_msg.append(f"负责人->{bulk_assignee}")
                            
                            if bulk_next_status != "(保持不变)":
                                update_params["status"] = bulk_next_status
                                log_msg.append(f"状态->{bulk_next_status}")
                            
                            if update_params:
                                if workorder_service.update_work_order(wid, **update_params):
                                    success_count += 1
                                    # 记录审计日志
                                    audit_service.log_action(user_info["id"], user_info["username"], "批量更新", "工单管理", 
                                                           f"批量更新工单 ID:{wid} ({', '.join(log_msg)})")
                        
                        if success_count > 0:
                            App.success(f"成功批量更新 {success_count} 条工单")
                            time.sleep(1)
                            App.rerun()
                        else:
                            App.warning("未选择任何更改内容")

        action_col, export_col, del_col = App.columns([1, 2, 2])

        with action_col:
            App.markdown("&nbsp;", unsafe_allow_html=True)

        with export_col:
            App.markdown("**📄 导出报告**")
            exp1, exp2 = App.columns(2)
            with exp1:
                if App.button("生成 Excel", key="wo_btn_gen_excel", use_container_width=True):
                    App.session_state["wo_export_xlsx"] = GetWorkOrderExcel(selected_rows).getvalue()
                if App.session_state.get("wo_export_xlsx"):
                    App.download_button(
                        "⬇ 下载 Excel",
                        data=App.session_state["wo_export_xlsx"],
                        file_name="workorders.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="wo_btn_dl_excel",
                        use_container_width=True,
                    )
            with exp2:
                if App.button("生成 PDF", key="wo_btn_gen_pdf", use_container_width=True):
                    App.session_state["wo_export_pdf"] = GetWorkOrderPDF(selected_rows).getvalue()
                if App.session_state.get("wo_export_pdf"):
                    App.download_button(
                        "⬇ 下载 PDF",
                        data=App.session_state["wo_export_pdf"],
                        file_name="workorders.pdf",
                        mime="application/pdf",
                        key="wo_btn_dl_pdf",
                        use_container_width=True,
                    )

        with del_col:
            App.markdown("**🗑️ 删除所选**")
            confirm = App.checkbox("确认删除（不可恢复）", key="workorder_del_confirm")
            if confirm:
                if App.button(f"删除 {len(selected_ids)} 条工单", key="workorder_del_btn", type="primary", use_container_width=True):
                    deleted = workorder_service.delete_work_orders(selected_ids)
                    App.success(f"已删除 {deleted} 条工单")
                    App.session_state.pop("workorder_del_confirm", None)
                    App.session_state.pop("wo_export_xlsx", None)
                    App.session_state.pop("wo_export_pdf", None)
    else:
        App.caption("勾选行可导出报告或删除工单")

    # 用原始 order_rows（未编辑的 id 列表）构建工单流转
    orig_ids = [row["id"] for row in order_rows]

    App.markdown("#### 🔄 工单流转")
    selected_order_id = App.selectbox("选择工单", orig_ids, key="workorder_selected_id",
        format_func=lambda x: next((f"{r['工单编号']} - {r['标题']}" for r in order_rows if r["id"] == x), str(x)))

    # 切换工单时重置输入框的 session_state，避免旧值残留
    prev_key = "workorder_prev_selected_id"
    if App.session_state.get(prev_key) != selected_order_id:
        App.session_state[prev_key] = selected_order_id
        for k in ["workorder_assignee", "workorder_notes", "workorder_next_status"]:
            if k in App.session_state:
                del App.session_state[k]

    current_order = next((item for item in order_rows if item["id"] == selected_order_id), None)
    if current_order:
        # 从数据库加载当前工单的完整 processing_notes（仅在未被用户编辑时初始化）
        if "workorder_notes" not in App.session_state:
            try:
                _session = workorder_service.db_manager.get_session()
                try:
                    from database import WorkOrder as _WO
                    _wo = _session.query(_WO).filter(_WO.id == selected_order_id).first()
                    App.session_state["workorder_notes"] = _wo.processing_notes or "" if _wo else ""
                finally:
                    workorder_service.db_manager.close_session(_session)
            except Exception:
                App.session_state["workorder_notes"] = ""

        if "workorder_assignee" not in App.session_state:
            App.session_state["workorder_assignee"] = current_order.get("负责人") or ""

        # ── 工单快速操作 (工人端) ──────────────────────────────
        user_info = App.session_state.get("user_info", {})
        if assignee_id and current_order:
            App.markdown("#### ⚡ 快速操作")
            q_col1, q_col2, q_col3 = App.columns(3)
            cur_s = current_order["状态"]
            
            if cur_s in ["已派单", "待确认"]:
                if q_col1.button("▶️ 开始维修", key="wo_start_btn", use_container_width=True, type="primary"):
                    workorder_service.update_work_order(selected_order_id, status="处理中", 
                                                       processing_notes=f"工人 {user_info['username']} 已开始现场维修。")
                    audit_service.log_action(user_info["id"], user_info["username"], "开始维修", "工单管理", f"工单 {current_order['工单编号']} 状态变更为：处理中")
                    App.rerun()
            
            if cur_s == "处理中":
                if q_col2.button("📸 提交现场验证", key="wo_verify_btn", use_container_width=True, type="primary"):
                    App.session_state["active_wo_for_detect"] = {
                        "id": selected_order_id,
                        "code": current_order["工单编号"],
                        "segment": current_order["管段"]
                    }
                    App.info("已选定工单，请前往【现场检测】上传维修后照片。")
            
            if cur_s == "已修复":
                App.success("🎉 该工单已提交修复结果，等待管理员复检。")

        update_col1, update_col2 = App.columns(2)
        with update_col1:
            status_options = ["待确认", "已派单", "处理中", "已修复", "已复检"]
            cur_status = current_order["状态"]
            next_status = App.selectbox(
                "更新状态",
                status_options,
                index=status_options.index(cur_status) if cur_status in status_options else 0,
                key="workorder_next_status",
            )
        with update_col2:
            if user_info.get("role") == "admin":
                # 管理员可以从下拉框选择工人
                all_workers = user_service.list_users(role="user", status="active")
                worker_names = [w.username for w in all_workers]
                if not worker_names:
                    App.warning("系统中暂无可用工人账号")
                    assignee = App.text_input("负责人 (手动输入)", key="workorder_assignee")
                else:
                    # 默认选中当前负责人，如果不在列表中则置空
                    try:
                        cur_idx = worker_names.index(current_order.get("负责人"))
                    except ValueError:
                        cur_idx = 0
                    assignee = App.selectbox("指派负责人", worker_names, index=cur_idx, key="workorder_assignee_sel")
            else:
                # 工人端固定为自己
                assignee = App.text_input("负责人", key="workorder_assignee", disabled=True)

        notes = App.text_area("处理记录", key="workorder_notes", placeholder="填写派单、维修、复检等过程说明")
        if App.button("💾 保存工单更新", key="workorder_update_btn", type="primary", use_container_width=True):
            # 如果是管理员使用了下拉框，则取下拉框的值
            final_assignee = assignee
            if user_info.get("role") == "admin" and 'workorder_assignee_sel' in App.session_state:
                final_assignee = App.session_state['workorder_assignee_sel']
            
            updated = workorder_service.update_work_order(
                selected_order_id,
                status=next_status,
                assignee=final_assignee,
                processing_notes=notes,
            )
            if updated:
                audit_service.log_action(user_info["id"], user_info["username"], "更新工单", "工单管理", 
                                       f"更新了工单 {current_order['工单编号']}：状态->{next_status}，负责人->{final_assignee}")
                App.success("工单已更新")
                App.rerun()
            else:
                App.warning("工单不存在或更新失败")

        App.markdown("#### 🧾 更新记录")
        logs = workorder_service.list_work_order_logs(selected_order_id, limit=300)
        if logs:
            log_rows = []
            for log in logs:
                log_rows.append(
                    {
                        "时间": log.created_at.strftime('%Y-%m-%d %H:%M:%S') if log.created_at else None,
                        "动作": log.action_type,
                        "状态变化": f"{log.old_status or '-'} -> {log.new_status or '-'}",
                        "负责人": log.assignee,
                        "备注": log.notes,
                    }
                )
            App.dataframe(log_rows, width="stretch", hide_index=True)
        else:
            App.info("暂无工单更新记录")


def show_admin_dispatch_center():
    """管理员调度中心：代办事项与全局概览"""
    App.markdown("### 🛰️ 调度指挥中心")
    
    # 获取统计数据
    stats = workorder_service.get_work_order_stats()
    status_dist = stats.get("status_distribution", {})
    
    # 代办提醒卡片
    c1, c2, c3 = App.columns(3)
    with c1:
        App.markdown(
            f"""<div class='task-card' style='border-left: 5px solid gray;'>
                <div style='font-size: 0.9rem; color: #666;'>待指派任务</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{status_dist.get("待确认", 0)}</div>
            </div>""", unsafe_allow_html=True
        )
    with c2:
        App.markdown(
            f"""<div class='task-card' style='border-left: 5px solid orange;'>
                <div style='font-size: 0.9rem; color: #666;'>正在维修中</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{status_dist.get("处理中", 0) + status_dist.get("已派单", 0)}</div>
            </div>""", unsafe_allow_html=True
        )
    with c3:
        App.markdown(
            f"""<div class='task-card' style='border-left: 5px solid green;'>
                <div style='font-size: 0.9rem; color: #666;'>待审核验收</div>
                <div style='font-size: 1.8rem; font-weight: bold;'>{status_dist.get("已修复", 0)}</div>
            </div>""", unsafe_allow_html=True
        )

    App.markdown("#### 🔔 紧急处理清单")
    # 获取需要关注的工单（待确认或已修复）
    pending_orders = workorder_service.list_work_orders(limit=10)
    urgent_orders = [o for o in pending_orders if o.status in ["待确认", "已修复", "处理中"]]
    
    if urgent_orders:
        for o in urgent_orders:
            with App.container(border=True):
                col_info, col_btn = App.columns([4, 1])
                with col_info:
                    status_label = "🔴 待指派" if o.status == "待确认" else "🟡 维修中" if o.status in ["已派单", "处理中"] else "🟢 待验收"
                    App.markdown(f"**{status_label} | {o.order_code}** - {o.title}")
                    App.caption(f"管段: {o.segment.segment_code if o.segment else '未知'} | 负责人: {o.assignee or '未指派'} | 更新: {o.updated_at.strftime('%m-%d %H:%M')}")
                with col_btn:
                    if App.button("处理", key=f"dispatch_{o.id}", use_container_width=True):
                        App.session_state["admin_active_tab"] = "🧰 工单管理"
                        # 同步 radio 组件的 key，确保可见选项也切换到目标 Tab
                        App.session_state["admin_tab_radio"] = "🧰 工单管理"
                        App.session_state["workorder_selected_id"] = o.id
                        App.rerun()
    else:
        App.info("目前暂无紧急待办事项，系统运行良好。")

@App.fragment
def show_worker_task_center(worker_id):
    """工人任务中心：今日任务清单"""
    user_info = App.session_state.get("user_info", {})
    App.markdown(f"### 📋 你好，{user_info['username']}")
    App.markdown("这是你今日的维修任务清单")

    # 获取该工人的任务
    my_tasks = workorder_service.list_work_orders(status="全部")
    my_tasks = [t for t in my_tasks if t.assignee == user_info["username"] and t.status != "已复检"]
    
    if not my_tasks:
        App.success("✨ 太棒了！你已经完成了所有分配的任务。")
        return

    for t in my_tasks:
        with App.container(border=True):
            c_info, c_action = App.columns([3, 1])
            with c_info:
                color = "blue" if t.status == "已派单" else "orange" if t.status == "处理中" else "green"
                App.markdown(f"<span style='color:{color};font-weight:bold;'>[{t.status}]</span> **{t.order_code}**", unsafe_allow_html=True)
                App.markdown(f"**任务:** {t.title}")
                App.caption(f"管段位置: {t.segment.location if t.segment else '未知'}")
            
            with c_action:
                if t.status in ["已派单", "待确认"]:
                    if App.button("▶️ 开始工作", key=f"start_task_{t.id}", type="primary", use_container_width=True):
                        workorder_service.update_work_order(t.id, status="处理中", processing_notes="工人已到达现场并开始维修。")
                        audit_service.log_action(user_info["id"], user_info["username"], "开工", "任务中心", f"开始维修工单 {t.order_code}")
                        App.rerun()
                elif t.status == "处理中":
                    if App.button("📸 拍照验收", key=f"verify_task_{t.id}", type="primary", use_container_width=True):
                        App.session_state["active_wo_for_detect"] = {
                            "id": t.id,
                            "code": t.order_code,
                            "segment": t.segment.segment_code if t.segment else "未知"
                        }
                        App.info("已关联工单，请切换到【现场验收】拍照。")
                elif t.status == "已修复":
                    App.markdown("<div style='text-align:center; color:green; margin-top:10px;'>等待审核</div>", unsafe_allow_html=True)

def show_settings_dashboard():
    """系统设置页"""
    App.markdown("### ⚙️ 系统设置")

    # ── 用户与审计管理 (仅管理员可见) ───────────────────────
    user_info = App.session_state.get("user_info", {})
    if user_info.get("role") == "admin":
        with App.expander("👥 人员管理与账号审批", expanded=True):
            users = user_service.list_users()
            user_rows = []
            for u in users:
                user_rows.append({
                    "ID": u.id,
                    "用户名": u.username,
                    "角色": "管理员" if u.role == "admin" else "现场工人",
                    "状态": "正常" if u.status == "active" else "待审批" if u.status == "pending" else "禁用",
                    "最后登录": u.last_login.strftime("%Y-%m-%d %H:%M") if u.last_login else "从未登录"
                })
            App.dataframe(user_rows, hide_index=True, use_container_width=True)
            
            # 审批操作
            pending_users = [u for u in users if u.status == "pending"]
            if pending_users:
                App.markdown("#### ⏳ 待处理申请")
                for pu in pending_users:
                    col1, col2, col3 = App.columns([2, 1, 1])
                    col1.write(f"用户: **{pu.username}** ({pu.email})")
                    if col2.button("通过", key=f"approve_{pu.id}"):
                        user_service.update_user_status(pu.id, "active")
                        audit_service.log_action(user_info["id"], user_info["username"], "审批", "用户管理", f"批准了用户 {pu.username} 的申请")
                        App.success(f"已批准 {pu.username}")
                        App.rerun()
                    if col3.button("拒绝", key=f"reject_{pu.id}"):
                        user_service.delete_user(pu.id)
                        audit_service.log_action(user_info["id"], user_info["username"], "审批", "用户管理", f"拒绝并删除了用户 {pu.username} 的申请")
                        App.rerun()

        with App.expander("📜 系统审计日志", expanded=False):
            logs = audit_service.list_logs(limit=200)
            if logs:
                log_rows = []
                for l in logs:
                    log_rows.append({
                        "时间": l.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "操作人": l.username,
                        "动作": l.action,
                        "模块": l.module,
                        "内容": l.content
                    })
                App.dataframe(log_rows, hide_index=True, use_container_width=True)
            else:
                App.info("暂无审计日志")
        App.markdown("---")

    # ── 风险阈值配置 ──────────────────────────────────────────
    App.markdown("#### 🎚️ 风险阈值配置")
    App.caption("变形率（Deformation）范围 0~1，值越大表示变形越严重")

    cur_high = float(db_ops.get_config("risk_threshold_high", 0.3))
    cur_mid  = float(db_ops.get_config("risk_threshold_mid",  0.15))

    th_col1, th_col2 = App.columns(2)
    with th_col1:
        new_high = App.slider(
            "高风险阈值（变形率 ≥ 此值 → 高风险）",
            min_value=0.05, max_value=1.0, value=cur_high, step=0.01,
            key="setting_risk_high",
        )
    with th_col2:
        new_mid = App.slider(
            "中风险阈值（变形率 ≥ 此值 → 中风险，低于此值 → 低风险）",
            min_value=0.01, max_value=1.0, value=cur_mid, step=0.01,
            key="setting_risk_mid",
        )

    if new_high <= new_mid:
        App.warning("高风险阈值必须大于中风险阈值")
    else:
        App.info(f"当前规则：变形率 ≥ {new_high:.2f} → 高风险 ｜ {new_mid:.2f} ~ {new_high:.2f} → 中风险 ｜ < {new_mid:.2f} → 低风险")
        if App.button("💾 保存风险阈值", key="setting_save_threshold"):
            db_ops.set_config("risk_threshold_high", new_high, "number", "高风险变形率阈值")
            db_ops.set_config("risk_threshold_mid",  new_mid,  "number", "中风险变形率阈值")
            App.success("风险阈值已保存")

    App.markdown("---")

    # ── 其他检测参数 ──────────────────────────────────────────
    App.markdown("#### 🔧 检测参数")

    cur_batch    = int(db_ops.get_config("batch_size_default", 5))
    cur_spacing  = int(db_ops.get_config("frame_spacing_default", 30))
    cur_mode     = str(db_ops.get_config("processing_mode_default", "逐帧"))

    p_col1, p_col2, p_col3 = App.columns(3)
    with p_col1:
        new_batch = App.number_input("默认批处理大小", min_value=1, max_value=50, value=cur_batch, step=1, key="setting_batch")
    with p_col2:
        new_spacing = App.number_input("默认帧间距（视频）", min_value=1, max_value=300, value=cur_spacing, step=1, key="setting_spacing")
    with p_col3:
        mode_options = ["逐帧", "间隔采样"]
        new_mode = App.selectbox("默认处理模式", mode_options,
            index=mode_options.index(cur_mode) if cur_mode in mode_options else 0,
            key="setting_mode")

    if App.button("💾 保存检测参数", key="setting_save_params"):
        db_ops.set_config("batch_size_default",      new_batch,   "number", "默认批处理大小")
        db_ops.set_config("frame_spacing_default",   new_spacing, "number", "默认帧间距")
        db_ops.set_config("processing_mode_default", new_mode,    "string", "默认处理模式")
        App.success("检测参数已保存")

    App.markdown("---")

    # ── 数据清理 ──────────────────────────────────────────────
    App.markdown("#### 🗑️ 历史数据清理")
    App.caption("清除指定天数之前的检测任务、结果及图像数据（工单数据不受影响）")

    clean_col1, clean_col2 = App.columns([2, 1])
    with clean_col1:
        days = App.slider("清除多少天前的数据", min_value=1, max_value=365, value=30, step=1, key="setting_clean_days")
        App.caption(f"将删除 {days} 天前创建的所有检测任务及其关联结果和图像")
    with clean_col2:
        App.markdown("<br>", unsafe_allow_html=True)
        confirm_clean = App.checkbox("确认清理（不可恢复）", key="setting_clean_confirm")
        if confirm_clean:
            if App.button("🗑️ 执行清理", key="setting_clean_btn", type="primary", use_container_width=True):
                with App.spinner("正在清理..."):
                    result = db_integration.cleanup_old_data(days)
                if result:
                    App.success(
                        f"清理完成：删除任务 {result['deleted_tasks']} 条，"
                        f"结果 {result['deleted_results']} 条，"
                        f"图像 {result['deleted_images']} 张"
                    )
                    App.session_state.pop("setting_clean_confirm", None)
                else:
                    App.info("没有符合条件的历史数据")

    App.markdown("---")

    # ── 当前配置总览 ──────────────────────────────────────────
    with App.expander("📋 当前所有配置项", expanded=False):
        try:
            from database import SystemConfig
            session = db_ops.db_manager.get_session()
            try:
                configs = session.query(SystemConfig).filter(SystemConfig.is_active == True).all()
                rows = [{"配置键": c.config_key, "值": c.config_value, "类型": c.config_type, "说明": c.description or ""} for c in configs]
                App.dataframe(rows, hide_index=True, use_container_width=True)
            finally:
                db_ops.db_manager.close_session(session)
        except Exception as e:
            App.error(f"读取配置失败: {e}")


@App.fragment
def show_pipeline_ledger():
    """管网电子档案管理页"""
    App.markdown("### 🗂️ 管网电子档案")
    
    user_info = App.session_state.get("user_info", {})

    segments = segment_service.list_segments()

    # ── 辅助：将任务 ID 列表压缩为范围字符串，如 [1,2,3,5,7,8] → "1-3, 5, 7-8" ──
    def _ids_to_range_str(ids):
        if not ids:
            return "—"
        ids = sorted(set(ids))
        parts, start, end = [], ids[0], ids[0]
        for i in ids[1:]:
            if i == end + 1:
                end = i
            else:
                parts.append(str(start) if start == end else f"{start}-{end}")
                start = end = i
        parts.append(str(start) if start == end else f"{start}-{end}")
        return ", ".join(parts)

    # ── 预查询每个管段的任务 ID 列表 ──────────────────────────
    def _build_seg_task_map():
        from database import DetectionTask as _DT
        _s = segment_service.db_manager.get_session()
        try:
            rows = _s.query(_DT.id, _DT.segment_id).filter(_DT.segment_id != None).all()
            mapping = {}
            for tid, sid in rows:
                mapping.setdefault(sid, []).append(tid)
            return mapping
        finally:
            segment_service.db_manager.close_session(_s)

    seg_task_map = _build_seg_task_map()

    # ── 台账列表 ──────────────────────────────────────────────
    if segments:
        rows = []
        for s in segments:
            task_ids = seg_task_map.get(s.id, [])
            rows.append({
                "选择": False,
                "id": s.id,
                "管段编号": s.segment_code,
                "位置": s.location or "",
                "管径(mm)": s.diameter,
                "材质": s.material or "",
                "长度(m)": s.length,
                "安装年份": s.install_year,
                "检测任务范围": _ids_to_range_str(task_ids),
                "任务数": len(task_ids),
                "上次检测": s.last_inspected_at.strftime('%Y-%m-%d') if s.last_inspected_at else "未检测",
                "备注": s.notes or "",
            })

        edited = App.data_editor(
            rows,
            column_config={
                "选择": App.column_config.CheckboxColumn("选择", default=False),
                "id": App.column_config.NumberColumn("ID", disabled=True),
                "管段编号": App.column_config.TextColumn("管段编号", disabled=True),
                "位置": App.column_config.TextColumn("位置", disabled=True),
                "管径(mm)": App.column_config.NumberColumn("管径(mm)", disabled=True),
                "材质": App.column_config.TextColumn("材质", disabled=True),
                "长度(m)": App.column_config.NumberColumn("长度(m)", disabled=True),
                "安装年份": App.column_config.NumberColumn("安装年份", disabled=True),
                "检测任务范围": App.column_config.TextColumn("检测任务范围", disabled=True, help="该管段下所有检测任务的 ID 范围，随分配操作自动更新"),
                "任务数": App.column_config.NumberColumn("任务数", disabled=True),
                "上次检测": App.column_config.TextColumn("上次检测", disabled=True),
                "备注": App.column_config.TextColumn("备注", disabled=True),
            },
            hide_index=True,
            use_container_width=True,
            key="segment_table_editor",
        )

        selected = [r for r in edited if r.get("选择")]
        if selected:
            App.caption(f"已选 {len(selected)} 条管段")
            # 只有管理员可以删除
            if user_info.get("role") == "admin":
                del_confirm = App.checkbox("确认删除所选管段（关联工单的管段关联将被清除）", key="seg_del_confirm")
                if del_confirm and App.button("🗑️ 删除所选管段", key="seg_del_btn", type="primary"):
                    for r in selected:
                        segment_service.delete_segment(r["id"])
                        audit_service.log_action(user_info["id"], user_info["username"], "删除", "管段台账", f"删除了管段 ID:{r['id']}")
                    App.success(f"已删除 {len(selected)} 条管段")
                    App.session_state.pop("seg_del_confirm", None)
                    App.rerun()
            else:
                App.info("💡 只有管理员有权删除管段。")
    else:
        App.info("暂无管段记录，请在下方新增")

    App.markdown("---")

    # ── 编辑已有管段 ──────────────────────────────────────────
    if segments:
        App.markdown("#### ✏️ 编辑管段信息")
        seg_options = {s.id: f"{s.segment_code} — {s.location or '无位置'}" for s in segments}
        edit_id = App.selectbox("选择要编辑的管段", list(seg_options.keys()),
                                format_func=lambda x: seg_options[x], key="seg_edit_select")
        cur = next((s for s in segments if s.id == edit_id), None)
        if cur:
            ec1, ec2, ec3 = App.columns(3)
            with ec1:
                e_loc = App.text_input("位置", value=cur.location or "", key="seg_e_loc")
                e_mat = App.text_input("材质", value=cur.material or "", key="seg_e_mat")
            with ec2:
                e_dia = App.number_input("管径(mm)", value=float(cur.diameter or 0), min_value=0.0, step=1.0, key="seg_e_dia")
                e_len = App.number_input("长度(m)", value=float(cur.length or 0), min_value=0.0, step=0.1, key="seg_e_len")
            with ec3:
                e_year = App.number_input("安装年份", value=int(cur.install_year or 2000), min_value=1900, max_value=2100, step=1, key="seg_e_year")
                e_notes = App.text_input("备注", value=cur.notes or "", key="seg_e_notes")

            if App.button("💾 保存编辑", key="seg_save_edit"):
                # 构建变更记录
                changes = []
                if e_loc != cur.location: changes.append(f"位置: {cur.location} -> {e_loc}")
                if e_dia != cur.diameter: changes.append(f"管径: {cur.diameter} -> {e_dia}")
                if e_len != cur.length: changes.append(f"长度: {cur.length} -> {e_len}")
                
                segment_service.update_segment(
                    edit_id,
                    location=e_loc, material=e_mat,
                    diameter=e_dia or None, length=e_len or None,
                    install_year=e_year or None, notes=e_notes,
                )
                audit_service.log_action(user_info["id"], user_info["username"], "修改", "管段台账", f"修改了管段 {cur.segment_code}。变更内容: {', '.join(changes) if changes else '无核心参数变更'}")
                App.success(f"管段 {cur.segment_code} 已更新")
                App.rerun()

        App.markdown("---")

    # ── 新增管段 ──────────────────────────────────────────────
    if user_info.get("role") == "admin":
        App.markdown("#### ➕ 新增管段")
        nc1, nc2, nc3 = App.columns(3)
        with nc1:
            n_code = App.text_input("管段编号（唯一）", placeholder="如 P-001", key="seg_n_code")
            n_loc  = App.text_input("位置描述", placeholder="如 XX路XX号附近", key="seg_n_loc")
        with nc2:
            n_dia  = App.number_input("管径(mm)", min_value=0.0, value=300.0, step=1.0, key="seg_n_dia")
            n_mat  = App.selectbox("材质", ["混凝土", "PVC", "钢管", "球墨铸铁", "陶瓷", "其他"], key="seg_n_mat")
        with nc3:
            n_len  = App.number_input("长度(m)", min_value=0.0, value=0.0, step=0.1, key="seg_n_len")
            n_year = App.number_input("安装年份", min_value=1900, max_value=2100, value=2010, step=1, key="seg_n_year")
        n_notes = App.text_input("备注", key="seg_n_notes")

        if App.button("➕ 新增管段", key="seg_add_btn"):
            if not n_code.strip():
                App.warning("管段编号不能为空")
            else:
                try:
                    segment_service.create_segment(
                        segment_code=n_code.strip(),
                        location=n_loc, diameter=n_dia or None,
                        material=n_mat, length=n_len or None,
                        install_year=n_year or None, notes=n_notes,
                    )
                    audit_service.log_action(user_info["id"], user_info["username"], "新增", "管段台账", f"新增了管段 {n_code}")
                    App.success(f"管段 {n_code} 已创建")
                    App.rerun()
                except Exception as e:
                    App.error(f"创建失败: {e}")
    else:
        App.info("💡 只有管理员有权新增管段台账。")

    if not segments:
        return

    App.markdown("---")

    # ── 任务 & 工单分配管理 ───────────────────────────────────
    App.markdown("#### 🔗 任务 / 工单分配管理")
    App.caption("将检测任务归属到对应管段，工单会随任务自动关联。支持 ID 范围批量分配或手动勾选分配。")

    assign_tab1, assign_tab2, assign_tab3 = App.tabs(["📊 当前分配总览", "⚡ 范围批量分配", "✋ 手动分配"])

    # ── 总览 ──────────────────────────────────────────────────
    with assign_tab1:
        from database import DetectionTask as _DT
        _sess = segment_service.db_manager.get_session()
        try:
            all_tasks = _sess.query(_DT).order_by(_DT.id).all()
            overview_rows = []
            for t in all_tasks:
                seg_code = ""
                if t.segment_id:
                    seg = next((s for s in segments if s.id == t.segment_id), None)
                    seg_code = seg.segment_code if seg else f"ID:{t.segment_id}"
                overview_rows.append({
                    "任务ID": t.id,
                    "任务名称": t.task_name,
                    "类型": t.task_type,
                    "状态": t.status,
                    "归属管段": seg_code or "未分配",
                    "创建时间": t.created_at.strftime('%Y-%m-%d') if t.created_at else "",
                })
        finally:
            segment_service.db_manager.close_session(_sess)

        if overview_rows:
            App.dataframe(overview_rows, hide_index=True, use_container_width=True)
            unassigned = sum(1 for r in overview_rows if r["归属管段"] == "未分配")
            App.caption(f"共 {len(overview_rows)} 条任务，其中 {unassigned} 条未分配")
        else:
            App.info("暂无检测任务")

    # ── 范围批量分配 ──────────────────────────────────────────
    with assign_tab2:
        App.markdown("按任务 ID 范围批量分配到指定管段，例如将 ID 1~50 的任务全部归属到 P-001。")
        seg_map = {s.id: f"{s.segment_code} — {s.location or '无位置'}" for s in segments}

        ra_col1, ra_col2, ra_col3 = App.columns(3)
        with ra_col1:
            ra_seg = App.selectbox("目标管段", list(seg_map.keys()),
                                   format_func=lambda x: seg_map[x], key="ra_seg")
        with ra_col2:
            ra_from = App.number_input("任务 ID 起始", min_value=1, value=1, step=1, key="ra_from")
        with ra_col3:
            ra_to = App.number_input("任务 ID 结束", min_value=1, value=50, step=1, key="ra_to")

        if ra_from > ra_to:
            App.warning("起始 ID 不能大于结束 ID")
        else:
            App.caption(f"将把 ID {ra_from} ~ {ra_to} 范围内的任务分配给「{seg_map[ra_seg]}」")
            if App.button("⚡ 执行范围分配", key="ra_exec_btn", use_container_width=True):
                cnt = segment_service.assign_task_id_range(ra_seg, ra_from, ra_to)
                segment_service.update_last_inspected(ra_seg)
                App.success(f"已将 {cnt} 条任务分配给 {seg_map[ra_seg]}")

        App.markdown("---")
        App.markdown("**取消范围内任务的管段分配**")
        un_col1, un_col2 = App.columns(2)
        with un_col1:
            un_from = App.number_input("任务 ID 起始", min_value=1, value=1, step=1, key="un_from")
        with un_col2:
            un_to = App.number_input("任务 ID 结束", min_value=1, value=10, step=1, key="un_to")
        if App.button("🔓 取消该范围分配", key="un_exec_btn"):
            _sess2 = segment_service.db_manager.get_session()
            try:
                from database import DetectionTask as _DT2
                updated = _sess2.query(_DT2).filter(
                    _DT2.id >= un_from, _DT2.id <= un_to
                ).update({_DT2.segment_id: None}, synchronize_session=False)
                _sess2.commit()
                App.success(f"已取消 {updated} 条任务的管段分配")
            except Exception as _e:
                _sess2.rollback()
                App.error(str(_e))
            finally:
                segment_service.db_manager.close_session(_sess2)

    # ── 手动分配 ──────────────────────────────────────────────
    with assign_tab3:
        App.markdown("选择管段后，勾选要分配的任务，点击分配按钮。")
        ma_seg = App.selectbox("目标管段", list(seg_map.keys()),
                               format_func=lambda x: seg_map[x], key="ma_seg")

        unassigned_tasks = segment_service.get_unassigned_tasks()
        assigned_tasks   = segment_service.get_tasks_by_segment(ma_seg)

        ma_col1, ma_col2 = App.columns(2)

        with ma_col1:
            App.markdown(f"**未分配任务（{len(unassigned_tasks)} 条）**")
            if unassigned_tasks:
                un_rows = [{"选择": False, "id": t.id, "任务名称": t.task_name,
                            "类型": t.task_type, "状态": t.status} for t in unassigned_tasks]
                un_edited = App.data_editor(
                    un_rows,
                    column_config={
                        "选择": App.column_config.CheckboxColumn("选择", default=False),
                        "id": App.column_config.NumberColumn("ID", disabled=True),
                        "任务名称": App.column_config.TextColumn("任务名称", disabled=True),
                        "类型": App.column_config.TextColumn("类型", disabled=True),
                        "状态": App.column_config.TextColumn("状态", disabled=True),
                    },
                    hide_index=True, use_container_width=True, key="ma_un_editor",
                )
                to_assign = [r["id"] for r in un_edited if r.get("选择")]
                if to_assign:
                    if App.button(f"➡ 分配 {len(to_assign)} 条到「{seg_map[ma_seg]}」",
                                  key="ma_assign_btn", use_container_width=True):
                        segment_service.assign_tasks_to_segment(to_assign, ma_seg)
                        segment_service.update_last_inspected(ma_seg)
                        App.success(f"已分配 {len(to_assign)} 条任务")
            else:
                App.info("所有任务均已分配")

        with ma_col2:
            App.markdown(f"**已分配到「{seg_map[ma_seg]}」的任务（{len(assigned_tasks)} 条）**")
            if assigned_tasks:
                as_rows = [{"选择": False, "id": t.id, "任务名称": t.task_name,
                            "类型": t.task_type, "状态": t.status} for t in assigned_tasks]
                as_edited = App.data_editor(
                    as_rows,
                    column_config={
                        "选择": App.column_config.CheckboxColumn("选择", default=False),
                        "id": App.column_config.NumberColumn("ID", disabled=True),
                        "任务名称": App.column_config.TextColumn("任务名称", disabled=True),
                        "类型": App.column_config.TextColumn("类型", disabled=True),
                        "状态": App.column_config.TextColumn("状态", disabled=True),
                    },
                    hide_index=True, use_container_width=True, key="ma_as_editor",
                )
                to_unassign = [r["id"] for r in as_edited if r.get("选择")]
                if to_unassign:
                    if App.button(f"⬅ 取消 {len(to_unassign)} 条的分配",
                                  key="ma_unassign_btn", use_container_width=True):
                        segment_service.unassign_tasks(to_unassign)
                        App.success(f"已取消 {len(to_unassign)} 条任务的分配")
            else:
                App.info("该管段暂无已分配任务")
