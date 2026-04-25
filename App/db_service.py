"""
数据库服务层
提供高级数据库操作接口
"""

from database import (
    DatabaseManager, DetectionTask, DetectionResult, ResultImage, 
    Model, User, WorkOrder, WorkOrderLog, PipelineSegment, SystemLog
)
from sqlalchemy import desc, func, and_, or_, case
from datetime import datetime, timedelta
import json
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from uuid import uuid4

class DetectionService:
    """检测服务类"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def create_task(self, task_name, task_type, input_file_path, user_id, model_id, parameters=None):
        """创建检测任务"""
        session = self.db_manager.get_session()
        try:
            # 获取文件大小
            import os
            file_size = os.path.getsize(input_file_path) if os.path.exists(input_file_path) else 0
            
            task = DetectionTask(
                task_name=task_name,
                task_type=task_type,
                input_file_path=input_file_path,
                input_file_size=file_size,
                user_id=user_id,
                model_id=model_id,
                parameters=json.dumps(parameters) if parameters else None,
                status='pending'
            )
            
            session.add(task)
            session.commit()
            return task.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def update_task_status(self, task_id, status, progress=None):
        """更新任务状态"""
        session = self.db_manager.get_session()
        try:
            task = session.query(DetectionTask).filter(DetectionTask.id == task_id).first()
            if task:
                task.status = status
                if progress is not None:
                    task.progress = progress
                
                if status == 'processing' and not task.started_at:
                    task.started_at = datetime.utcnow()
                elif status in ['completed', 'failed']:
                    task.completed_at = datetime.utcnow()
                
                session.commit()
                
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def save_detection_result(self, task_id, frame_number, analysis_data, images_data):
        """保存检测结果"""
        session = self.db_manager.get_session()
        try:
            # 创建检测结果记录
            result = DetectionResult(
                task_id=task_id,
                frame_number=frame_number,
                shape=analysis_data.get('Shape', [None])[0] if isinstance(analysis_data.get('Shape'), list) else analysis_data.get('Shape'),
                aspect_ratio=analysis_data.get('AspectRatio', [None])[0] if isinstance(analysis_data.get('AspectRatio'), list) else analysis_data.get('AspectRatio'),
                orientation=analysis_data.get('Orientation', [None])[0] if isinstance(analysis_data.get('Orientation'), list) else analysis_data.get('Orientation'),
                deformation=analysis_data.get('Deformation', [None])[0] if isinstance(analysis_data.get('Deformation'), list) else analysis_data.get('Deformation'),
                conclusion=analysis_data.get('Conclusion', [None])[0] if isinstance(analysis_data.get('Conclusion'), list) else analysis_data.get('Conclusion'),
                result_data=json.dumps(analysis_data),
                created_at=datetime.utcnow()
            )
            
            session.add(result)
            session.flush()  # 获取result.id
            
            # 保存图像数据
            for image_type, image_array in images_data.items():
                if image_array is not None:
                    # 转换numpy数组为PIL图像
                    if len(image_array.shape) == 3:
                        if image_array.shape[2] == 3:  # RGB
                            pil_image = Image.fromarray(image_array)
                        elif image_array.shape[2] == 4:  # RGBA
                            pil_image = Image.fromarray(image_array)
                        else:
                            continue
                    elif len(image_array.shape) == 2:  # 灰度图
                        pil_image = Image.fromarray(image_array, mode='L')
                    else:
                        continue
                    
                    # 转换为字节数据
                    img_buffer = BytesIO()
                    pil_image.save(img_buffer, format='PNG')
                    img_data = img_buffer.getvalue()
                    
                    result_image = ResultImage(
                        result_id=result.id,
                        image_type=image_type,
                        image_data=img_data,
                        image_format='PNG',
                        width=image_array.shape[1],
                        height=image_array.shape[0],
                        file_size=len(img_data)
                    )
                    
                    session.add(result_image)
            
            session.commit()
            return result.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def get_task_results(self, task_id):
        """获取任务结果"""
        session = self.db_manager.get_session()
        try:
            results = session.query(DetectionResult).filter(
                DetectionResult.task_id == task_id
            ).order_by(DetectionResult.frame_number).all()
            
            return results
            
        finally:
            self.db_manager.close_session(session)
    
    def get_result_images(self, result_id):
        """获取结果图像"""
        session = self.db_manager.get_session()
        try:
            images = session.query(ResultImage).filter(
                ResultImage.result_id == result_id
            ).all()
            
            return images
            
        finally:
            self.db_manager.close_session(session)

class ModelService:
    """模型服务类"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def register_model(self, name, version, model_type, file_path, user_id, description=""):
        """注册模型"""
        session = self.db_manager.get_session()
        try:
            import os
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            model = Model(
                name=name,
                version=version,
                model_type=model_type,
                file_path=file_path,
                file_size=file_size,
                description=description,
                user_id=user_id,
                is_active=True
            )
            
            session.add(model)
            session.commit()
            return model.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def get_active_models(self, user_id=None):
        """获取活跃模型"""
        session = self.db_manager.get_session()
        try:
            query = session.query(Model).filter(Model.is_active == True)
            if user_id:
                query = query.filter(Model.user_id == user_id)
            
            return query.order_by(desc(Model.created_at)).all()
            
        finally:
            self.db_manager.close_session(session)

class StatisticsService:
    """统计服务类"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def get_task_statistics(self, days=30):
        """获取任务统计"""
        session = self.db_manager.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # 总任务数
            total_tasks = session.query(DetectionTask).filter(
                DetectionTask.created_at >= start_date
            ).count()
            
            # 完成任务数
            completed_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.status == 'completed'
                )
            ).count()
            
            # 失败任务数
            failed_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.status == 'failed'
                )
            ).count()
            
            # 处理的任务类型统计
            image_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.task_type == 'image'
                )
            ).count()
            
            video_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.task_type == 'video'
                )
            ).count()
            
            # 检测结果统计
            total_results = session.query(DetectionResult).filter(
                DetectionResult.created_at >= start_date
            ).count()
            
            # 形状分布统计
            shape_stats = session.query(
                DetectionResult.shape,
                func.count(DetectionResult.id).label('count')
            ).filter(
                DetectionResult.created_at >= start_date
            ).group_by(DetectionResult.shape).all()
            
            return {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
                'image_tasks': image_tasks,
                'video_tasks': video_tasks,
                'total_results': total_results,
                'shape_distribution': {shape: count for shape, count in shape_stats}
            }
            
        finally:
            self.db_manager.close_session(session)
    
    def get_performance_metrics(self, days=7):
        """获取性能指标"""
        session = self.db_manager.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # 平均处理时间
            completed_tasks = session.query(DetectionTask).filter(
                and_(
                    DetectionTask.created_at >= start_date,
                    DetectionTask.status == 'completed',
                    DetectionTask.started_at.isnot(None),
                    DetectionTask.completed_at.isnot(None)
                )
            ).all()
            
            if completed_tasks:
                total_time = sum([
                    (task.completed_at - task.started_at).total_seconds()
                    for task in completed_tasks
                ])
                avg_processing_time = total_time / len(completed_tasks)
            else:
                avg_processing_time = 0
            
            # 每日任务统计
            daily_stats = session.query(
                func.date(DetectionTask.created_at).label('date'),
                func.count(DetectionTask.id).label('task_count'),
                func.sum(case((DetectionTask.status == 'completed', 1), else_=0)).label('completed_count')
            ).filter(
                DetectionTask.created_at >= start_date
            ).group_by(func.date(DetectionTask.created_at)).all()
            
            return {
                'average_processing_time': avg_processing_time,
                'daily_statistics': [
                    {
                        'date': stat.date.strftime('%Y-%m-%d') if hasattr(stat.date, 'strftime') else str(stat.date),
                        'total_tasks': stat.task_count,
                        'completed_tasks': stat.completed_count
                    }
                    for stat in daily_stats
                ]
            }
            
        finally:
            self.db_manager.close_session(session)

class UserService:
    """用户服务类"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def create_user(self, username, email, password_hash, role='user'):
        """创建用户"""
        session = self.db_manager.get_session()
        try:
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role
            )
            
            session.add(user)
            session.commit()
            return user.id
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def get_user_by_username(self, username):
        """根据用户名获取用户"""
        session = self.db_manager.get_session()
        try:
            return session.query(User).filter(User.username == username).first()
        finally:
            self.db_manager.close_session(session)

    def verify_user(self, username, password):
        """验证用户登录 (简单明文验证，实际项目应使用哈希)"""
        user = self.get_user_by_username(username)
        if user and user.password_hash == password:  # 生产环境应使用 check_password_hash
            return user
        return None

    def update_last_login(self, user_id):
        """更新最后登录时间"""
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def list_users(self, role=None, status=None):
        """列出用户"""
        session = self.db_manager.get_session()
        try:
            query = session.query(User)
            if role:
                query = query.filter(User.role == role)
            if status:
                query = query.filter(User.status == status)
            return query.all()
        finally:
            self.db_manager.close_session(session)

    def update_user_status(self, user_id, status):
        """更新用户状态（审批账号）"""
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.status = status
                session.commit()
                return True
            return False
        except Exception:
            session.rollback()
            return False
        finally:
            self.db_manager.close_session(session)

    def delete_user(self, user_id):
        """删除用户"""
        session = self.db_manager.get_session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if user and user.role != 'admin': # 禁止删除管理员
                session.delete(user)
                session.commit()
                return True
            return False
        except Exception:
            session.rollback()
            return False
        finally:
            self.db_manager.close_session(session)

class AuditService:
    """系统审计服务类"""
    def __init__(self):
        self.db_manager = DatabaseManager()

    def log_action(self, user_id, username, action, module, content, ip_address=None):
        """记录审计日志"""
        session = self.db_manager.get_session()
        try:
            log = SystemLog(
                user_id=user_id,
                username=username,
                action=action,
                module=module,
                content=content,
                ip_address=ip_address
            )
            session.add(log)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"审计日志记录失败: {e}")
        finally:
            self.db_manager.close_session(session)

    def list_logs(self, limit=500):
        """获取审计日志"""
        session = self.db_manager.get_session()
        try:
            return session.query(SystemLog).order_by(desc(SystemLog.created_at)).limit(limit).all()
        finally:
            self.db_manager.close_session(session)

class WorkOrderService:
    """工单服务类"""

    STATUS_FLOW = ["待确认", "已派单", "处理中", "已修复", "已复检"]

    def __init__(self):
        self.db_manager = DatabaseManager()

    def _add_order_log(self, session, work_order_id, action_type, old_status=None, new_status=None, assignee=None, notes=None):
        log = WorkOrderLog(
            work_order_id=work_order_id,
            action_type=action_type,
            old_status=old_status,
            new_status=new_status,
            assignee=assignee,
            notes=notes,
        )
        session.add(log)

    def _build_order_code(self, session=None):
        for _ in range(8):
            code = f"WO-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}-{uuid4().hex[:4].upper()}"
            if session is None:
                return code
            exists = session.query(WorkOrder).filter(WorkOrder.order_code == code).first()
            if not exists:
                return code
        return f"WO-{uuid4().hex.upper()}"

    def _risk_to_priority(self, risk_level):
        if risk_level == "高":
            return "高"
        if risk_level == "低":
            return "低"
        return "中"

    def create_work_order(self, title, risk_level="中", task_id=None, result_id=None, segment_id=None, assignee=None, issue_description=None, action_suggestion=None):
        session = self.db_manager.get_session()
        try:
            order = WorkOrder(
                order_code=self._build_order_code(session),
                title=title,
                risk_level=risk_level or "中",
                priority=self._risk_to_priority(risk_level),
                status="待确认",
                assignee=assignee,
                issue_description=issue_description,
                action_suggestion=action_suggestion,
                task_id=task_id,
                result_id=result_id,
                segment_id=segment_id,
            )
            session.add(order)
            session.flush()
            self._add_order_log(
                session,
                work_order_id=order.id,
                action_type="创建",
                old_status=None,
                new_status=order.status,
                assignee=order.assignee,
                notes=order.issue_description,
            )
            session.commit()
            return order.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def create_work_orders_from_task(self, task_id, risk_level="高", segment_id=None):
        session = self.db_manager.get_session()
        try:
            results = session.query(DetectionResult).filter(
                DetectionResult.task_id == task_id
            ).order_by(DetectionResult.frame_number).all()

            result_ids = [result.id for result in results]
            existing_result_ids = set()
            if result_ids:
                existing_rows = session.query(WorkOrder.result_id).filter(
                    WorkOrder.result_id.in_(result_ids)
                ).all()
                existing_result_ids = {row[0] for row in existing_rows if row[0] is not None}

            created_count = 0
            for result in results:
                if result.id in existing_result_ids:
                    continue

                parsed = {}
                try:
                    parsed = json.loads(result.result_data) if result.result_data else {}
                except Exception:
                    parsed = {}

                current_risk = parsed.get("RiskLevel")
                action_suggestion = parsed.get("ActionSuggestion")
                if isinstance(current_risk, list):
                    current_risk = current_risk[0] if current_risk else None
                if isinstance(action_suggestion, list):
                    action_suggestion = action_suggestion[0] if action_suggestion else None

                if current_risk != risk_level:
                    continue

                frame_number = result.frame_number if result.frame_number is not None else "未知"
                title = f"任务{task_id}-帧{frame_number}风险处置"
                order = WorkOrder(
                    order_code=self._build_order_code(session),
                    title=title,
                    risk_level=current_risk,
                    priority=self._risk_to_priority(current_risk),
                    status="待确认",
                    issue_description=f"检测结论: {result.conclusion}，形状: {result.shape}，变形: {result.deformation}",
                    action_suggestion=action_suggestion,
                    task_id=task_id,
                    result_id=result.id,
                    segment_id=segment_id,
                )
                session.add(order)
                session.flush()
                self._add_order_log(
                    session,
                    work_order_id=order.id,
                    action_type="创建",
                    old_status=None,
                    new_status=order.status,
                    assignee=order.assignee,
                    notes=order.action_suggestion,
                )
                existing_result_ids.add(result.id)
                created_count += 1

            session.commit()
            return created_count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def list_work_orders(self, status=None, limit=200):
        from sqlalchemy.orm import joinedload
        session = self.db_manager.get_session()
        try:
            query = session.query(WorkOrder).options(joinedload(WorkOrder.segment))
            if status and status != "全部":
                query = query.filter(WorkOrder.status == status)
            return query.order_by(desc(WorkOrder.created_at)).limit(limit).all()
        finally:
            self.db_manager.close_session(session)

    def update_work_order(self, order_id, status=None, assignee=None, processing_notes=None):
        session = self.db_manager.get_session()
        try:
            order = session.query(WorkOrder).filter(WorkOrder.id == order_id).first()
            if not order:
                return False

            old_status = order.status

            if status:
                order.status = status
                now = datetime.utcnow()
                if status == "已派单" and order.dispatched_at is None:
                    order.dispatched_at = now
                elif status == "处理中" and order.processing_at is None:
                    order.processing_at = now
                elif status == "已修复" and order.repaired_at is None:
                    order.repaired_at = now
                elif status == "已复检" and order.rechecked_at is None:
                    order.rechecked_at = now

            if assignee is not None:
                order.assignee = assignee
            if processing_notes is not None:
                order.processing_notes = processing_notes

            self._add_order_log(
                session,
                work_order_id=order.id,
                action_type="更新",
                old_status=old_status,
                new_status=order.status,
                assignee=order.assignee,
                notes=order.processing_notes,
            )

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def get_work_order_stats(self, assignee_id=None):
        session = self.db_manager.get_session()
        try:
            query = session.query(WorkOrder)
            if assignee_id:
                # 获取用户名进行过滤
                user = session.query(User).filter(User.id == assignee_id).first()
                if user:
                    query = query.filter(WorkOrder.assignee == user.username)
            
            total = query.count()
            by_status = session.query(
                WorkOrder.status,
                func.count(WorkOrder.id).label('count')
            )
            if assignee_id:
                user = session.query(User).filter(User.id == assignee_id).first()
                if user:
                    by_status = by_status.filter(WorkOrder.assignee == user.username)
            
            by_status = by_status.group_by(WorkOrder.status).all()
            
            closed_query = session.query(WorkOrder).filter(WorkOrder.status == "已复检")
            if assignee_id:
                user = session.query(User).filter(User.id == assignee_id).first()
                if user:
                    closed_query = closed_query.filter(WorkOrder.assignee == user.username)
            
            closed = closed_query.count()
            return {
                "total": total,
                "closed": closed,
                "closed_rate": (closed / total * 100) if total > 0 else 0,
                "status_distribution": {status: count for status, count in by_status},
            }
        finally:
            self.db_manager.close_session(session)

    def list_work_order_logs(self, work_order_id, limit=200):
        session = self.db_manager.get_session()
        try:
            return session.query(WorkOrderLog).filter(
                WorkOrderLog.work_order_id == work_order_id
            ).order_by(desc(WorkOrderLog.created_at)).limit(limit).all()
        finally:
            self.db_manager.close_session(session)

    def delete_work_orders(self, order_ids):
        """批量删除工单及其日志，返回实际删除数量"""
        if not order_ids:
            return 0
        session = self.db_manager.get_session()
        try:
            session.query(WorkOrderLog).filter(WorkOrderLog.work_order_id.in_(order_ids)).delete(synchronize_session=False)
            deleted = session.query(WorkOrder).filter(WorkOrder.id.in_(order_ids)).delete(synchronize_session=False)
            session.commit()
            return deleted
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)


class PipelineSegmentService:
    """管段台账服务"""

    def __init__(self):
        self.db_manager = DatabaseManager()

    def list_segments(self):
        session = self.db_manager.get_session()
        try:
            return session.query(PipelineSegment).order_by(PipelineSegment.segment_code).all()
        finally:
            self.db_manager.close_session(session)

    def get_segment(self, segment_id):
        session = self.db_manager.get_session()
        try:
            return session.query(PipelineSegment).filter(PipelineSegment.id == segment_id).first()
        finally:
            self.db_manager.close_session(session)

    def create_segment(self, segment_code, location="", diameter=None, material="", length=None, install_year=None, notes=""):
        session = self.db_manager.get_session()
        try:
            exists = session.query(PipelineSegment).filter(PipelineSegment.segment_code == segment_code).first()
            if exists:
                raise ValueError(f"管段编号 {segment_code} 已存在")
            seg = PipelineSegment(
                segment_code=segment_code,
                location=location,
                diameter=diameter,
                material=material,
                length=length,
                install_year=install_year,
                notes=notes,
            )
            session.add(seg)
            session.commit()
            return seg.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def update_segment(self, segment_id, **kwargs):
        session = self.db_manager.get_session()
        try:
            seg = session.query(PipelineSegment).filter(PipelineSegment.id == segment_id).first()
            if not seg:
                return False
            for k, v in kwargs.items():
                if hasattr(seg, k):
                    setattr(seg, k, v)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def delete_segment(self, segment_id):
        session = self.db_manager.get_session()
        try:
            seg = session.query(PipelineSegment).filter(PipelineSegment.id == segment_id).first()
            if not seg:
                return False
            session.delete(seg)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def update_last_inspected(self, segment_id, dt=None):
        return self.update_segment(segment_id, last_inspected_at=dt or datetime.utcnow())

    def assign_tasks_to_segment(self, task_ids, segment_id):
        """将一批任务分配给指定管段"""
        session = self.db_manager.get_session()
        try:
            session.query(DetectionTask).filter(
                DetectionTask.id.in_(task_ids)
            ).update({DetectionTask.segment_id: segment_id}, synchronize_session=False)
            session.commit()
            return len(task_ids)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def unassign_tasks(self, task_ids):
        """取消任务的管段分配"""
        session = self.db_manager.get_session()
        try:
            session.query(DetectionTask).filter(
                DetectionTask.id.in_(task_ids)
            ).update({DetectionTask.segment_id: None}, synchronize_session=False)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)

    def get_tasks_by_segment(self, segment_id):
        """获取某管段下的所有任务"""
        from sqlalchemy.orm import joinedload
        session = self.db_manager.get_session()
        try:
            return session.query(DetectionTask).filter(
                DetectionTask.segment_id == segment_id
            ).order_by(DetectionTask.id).all()
        finally:
            self.db_manager.close_session(session)

    def get_unassigned_tasks(self):
        """获取未分配管段的任务"""
        session = self.db_manager.get_session()
        try:
            return session.query(DetectionTask).filter(
                DetectionTask.segment_id == None
            ).order_by(DetectionTask.id).all()
        finally:
            self.db_manager.close_session(session)

    def assign_task_id_range(self, segment_id, id_from, id_to):
        """按 ID 范围批量分配任务到管段"""
        session = self.db_manager.get_session()
        try:
            updated = session.query(DetectionTask).filter(
                DetectionTask.id >= id_from,
                DetectionTask.id <= id_to,
            ).update({DetectionTask.segment_id: segment_id}, synchronize_session=False)
            session.commit()
            return updated
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)


# 全局服务实例
detection_service = DetectionService()
model_service = ModelService()
statistics_service = StatisticsService()
user_service = UserService()
audit_service = AuditService()
workorder_service = WorkOrderService()
segment_service = PipelineSegmentService()
