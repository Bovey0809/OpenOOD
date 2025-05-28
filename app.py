#!/usr/bin/env python3
"""
种子分类和OOD检测Web应用
提供用户友好的Web界面，支持图像上传和结果展示
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
import torch

# 导入我们的推理管道
from scripts.simple_inference_pipeline import SimpleInferencePipeline, DetectionResult

app = Flask(__name__)
app.secret_key = 'seed_ood_detection_secret_key_2024'

# 配置
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 创建必要的目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# 全局变量存储推理管道
inference_pipeline = None

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_pipeline():
    """初始化推理管道"""
    global inference_pipeline
    try:
        inference_pipeline = SimpleInferencePipeline()
        print("✅ 推理管道初始化成功")
        return True
    except Exception as e:
        print(f"❌ 推理管道初始化失败: {e}")
        return False

def format_confidence(confidence):
    """格式化置信度显示"""
    return f"{confidence:.1%}"

def get_confidence_color(confidence):
    """根据置信度返回颜色类"""
    if confidence >= 0.8:
        return "success"
    elif confidence >= 0.6:
        return "warning"
    else:
        return "danger"

def get_ood_badge(is_ood):
    """获取OOD标识"""
    if is_ood:
        return {"text": "外来种子", "class": "danger"}
    else:
        return {"text": "已知种子", "class": "success"}

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """文件上传页面"""
    if request.method == 'POST':
        # 检查是否有文件
        if 'files' not in request.files:
            flash('没有选择文件', 'error')
            return redirect(request.url)
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            flash('没有选择文件', 'error')
            return redirect(request.url)
        
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # 添加时间戳避免文件名冲突
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{timestamp}{ext}"
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filename)
        
        if uploaded_files:
            flash(f'成功上传 {len(uploaded_files)} 个文件', 'success')
            return redirect(url_for('detect', files=','.join(uploaded_files)))
        else:
            flash('没有有效的图像文件', 'error')
    
    return render_template('upload.html')

@app.route('/detect')
def detect():
    """检测页面"""
    files = request.args.get('files', '').split(',')
    files = [f for f in files if f]  # 过滤空字符串
    
    if not files:
        flash('没有要检测的文件', 'error')
        return redirect(url_for('upload_file'))
    
    return render_template('detect.html', files=files)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API检测接口"""
    global inference_pipeline
    
    if inference_pipeline is None:
        return jsonify({'error': '推理管道未初始化'}), 500
    
    data = request.get_json()
    files = data.get('files', [])
    
    if not files:
        return jsonify({'error': '没有要检测的文件'}), 400
    
    try:
        # 构建文件路径
        file_paths = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in files]
        
        # 检查文件是否存在
        missing_files = [f for f, path in zip(files, file_paths) if not os.path.exists(path)]
        if missing_files:
            return jsonify({'error': f'文件不存在: {missing_files}'}), 400
        
        # 执行检测
        results = inference_pipeline.detect_batch(file_paths)
        
        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_result = {
                'filename': result.filename,
                'predicted_class': result.predicted_class,
                'predicted_class_name': result.predicted_class_name,
                'confidence': result.confidence,
                'confidence_formatted': format_confidence(result.confidence),
                'confidence_color': get_confidence_color(result.confidence),
                'is_ood': result.is_ood,
                'ood_badge': get_ood_badge(result.is_ood),
                'probabilities': result.probabilities,
                'processing_time': result.processing_time,
                'error': result.error
            }
            formatted_results.append(formatted_result)
        
        # 计算统计信息
        successful_results = [r for r in results if r.error is None]
        stats = {
            'total_images': len(results),
            'successful_images': len(successful_results),
            'failed_images': len(results) - len(successful_results),
            'id_samples': sum(1 for r in successful_results if not r.is_ood),
            'ood_samples': sum(1 for r in successful_results if r.is_ood),
            'avg_processing_time': sum(r.processing_time for r in successful_results) / len(successful_results) if successful_results else 0
        }
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"detection_results_{timestamp}.json"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_file)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'stats': stats,
                'results': [r.to_dict() for r in results]
            }, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'results': formatted_results,
            'result_file': result_file
        })
        
    except Exception as e:
        return jsonify({'error': f'检测失败: {str(e)}'}), 500

@app.route('/results')
def results():
    """结果历史页面"""
    result_files = []
    results_dir = Path(app.config['RESULTS_FOLDER'])
    
    if results_dir.exists():
        for file_path in results_dir.glob('detection_results_*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result_files.append({
                    'filename': file_path.name,
                    'timestamp': data.get('timestamp', ''),
                    'stats': data.get('stats', {}),
                    'size': file_path.stat().st_size
                })
            except:
                continue
    
    # 按时间戳排序
    result_files.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('results.html', result_files=result_files)

@app.route('/results/<filename>')
def view_result(filename):
    """查看具体结果"""
    result_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if not os.path.exists(result_path):
        flash('结果文件不存在', 'error')
        return redirect(url_for('results'))
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 格式化结果用于显示
        formatted_results = []
        for result in data.get('results', []):
            formatted_result = result.copy()
            formatted_result['confidence_formatted'] = format_confidence(result['confidence'])
            formatted_result['confidence_color'] = get_confidence_color(result['confidence'])
            formatted_result['ood_badge'] = get_ood_badge(result['is_ood'])
            formatted_results.append(formatted_result)
        
        return render_template('view_result.html', 
                             data=data, 
                             results=formatted_results,
                             filename=filename)
    
    except Exception as e:
        flash(f'读取结果文件失败: {e}', 'error')
        return redirect(url_for('results'))

@app.route('/download/<filename>')
def download_result(filename):
    """下载结果文件"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """访问上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/status')
def api_status():
    """API状态检查"""
    global inference_pipeline
    
    status = {
        'pipeline_ready': inference_pipeline is not None,
        'cuda_available': torch.cuda.is_available(),
        'device': str(inference_pipeline.device) if inference_pipeline else 'N/A'
    }
    
    if torch.cuda.is_available():
        status['gpu_count'] = torch.cuda.device_count()
        status['gpu_name'] = torch.cuda.get_device_name(0)
    
    return jsonify(status)

# 错误处理
@app.errorhandler(413)
def too_large(e):
    flash('文件太大，请选择小于16MB的文件', 'error')
    return redirect(url_for('upload_file'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# 模板过滤器
@app.template_filter('filesize')
def filesize_filter(size):
    """文件大小格式化"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

if __name__ == '__main__':
    print("🚀 启动种子分类和OOD检测Web应用...")
    
    # 初始化推理管道
    if init_pipeline():
        print("🌐 启动Web服务器...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ 无法启动应用，推理管道初始化失败") 