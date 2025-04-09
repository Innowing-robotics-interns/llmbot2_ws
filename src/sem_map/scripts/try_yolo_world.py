import sys
print(f"Python版本：{sys.version}")
import os
repo_path = "~/YOLO-World"  # 仓库克隆路径
image_path = "~/Pictures/TestSeg/BillieDesktop.jpg"   # 替换为您的图片路径
output_dir = "./outputs"    # 结果保存目录
os.makedirs(output_dir, exist_ok=True)
from mmyolo.apis import DetInferencer
import torch

class YOLOWorldDetector:
    def __init__(self, model_size='s'):
        self.configs = {
            's': ('configs/pretrain/yolo_world_v2_s.py', 
                 'https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_s.pth'),
            'm': ('configs/pretrain/yolo_world_v2_m.py',
                 'https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_m.pth')
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._check_weights()
        self.inferencer = self._build_inferencer(model_size)
  
    def _check_weights(self):
        """检查权重文件是否存在"""
        if not os.path.exists("weights"):
            os.makedirs("weights", exist_ok=True)
            print("正在自动下载预训练权重...")
          
    def _build_inferencer(self, model_size):
        config, checkpoint = self.configs[model_size]
        return DetInferencer(
            model=config,
            weights=checkpoint,
            device=self.device,
            pred_score_thr=0.4  # 置信度阈值
        )

# 初始化检测器（可选's'或'm'）
detector = YOLOWorldDetector(model_size='s')
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class ResultVisualizer:
    @staticmethod
    def plot_results(image_path, results):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
      
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')
      
        for det in results['predictions'][0]:
            label = det['label']
            conf = det['confidence']
            bbox = det['bbox']
          
            if conf < 0.3:  # 可视化阈值
                continue
              
            x1, y1, x2, y2 = map(int, bbox)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=1, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{label} {conf:.2f}", 
                    color='white', fontsize=10, 
                    bbox=dict(facecolor='lime', alpha=0.7))
      
        plt.show()
  
    @staticmethod
    def save_results(image_path, results):
        base_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"detected_{base_name}")
        cv2.imwrite(save_path, cv2.cvtColor(results['visualization'][0], cv2.COLOR_RGB2BGR))
        print(f"结果已保存至：{save_path}")

def run_detection(text_prompts=['person', 'car', 'dog']):
    # 输入验证
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片路径不存在：{image_path}")
  
    # 执行推理
    results = detector.inferencer(
        image_path,
        texts=[text_prompts],  # 支持多组文本提示
        return_vis=True
    )
  
    # 显示结果
    clear_output(wait=True)
    print("检测统计：")
    print(f"- 总检测数：{len(results['predictions'][0])}")
    print(f"- 使用设备：{detector.device.upper()}")
    print(f"- 推理时间：{results['time'][0]:.2f}s\n")
  
    # 可视化
    ResultVisualizer.plot_results(image_path, results)
    ResultVisualizer.save_results(image_path, results)
  
    return results

if __name__ == "__main__":
    # 自定义检测目标（修改此处）
    custom_labels = ['cell phone', 'laptop', 'book']  # 示例目标
  
    # 执行检测
    detection_results = run_detection(text_prompts=custom_labels)
  
    # 打印详细结果
    print("\n详细检测结果：")
    for i, det in enumerate(detection_results['predictions'][0]):
        if det['confidence'] > 0.3:
            print(f"目标{i+1}: {det['label']} | 置信度: {det['confidence']:.2f} | 位置: {list(map(int, det['bbox']))}")