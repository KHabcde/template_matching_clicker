import cv2
import numpy as np
import pyautogui
import sys
import os

class TemplateMatchingClicker:
    def __init__(self, template_path, threshold=0.8):
        """
        Parameters:
        -----------
        template_path : str
            テンプレート画像のファイルパス
        threshold : float
            マッチングの閾値 (0.0 ~ 1.0)
        """
        self.template_path = template_path
        self.threshold = threshold
        self.template_image = None

    def load_template(self):
        """テンプレート画像を読み込む"""
        if not os.path.exists(self.template_path):
            print(f"エラー: テンプレート画像が見つかりません: {self.template_path}")
            return False

        self.template_image = cv2.imread(self.template_path)
        if self.template_image is None:
            print(f"エラー: テンプレート画像の読み込みに失敗しました: {self.template_path}")
            return False

        print(f"テンプレート画像を読み込みました: {self.template_path}")
        return True

    def take_screenshot(self):
        """スクリーンショットを取得しOpenCV形式に変換"""
        try:
            screenshot = pyautogui.screenshot()
            # PILのRGBA形式からOpenCVのBGR形式に変換
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            print("スクリーンショットを取得しました")
            return screenshot_bgr
        except Exception as e:
            print(f"エラー: スクリーンショットの取得に失敗しました: {str(e)}")
            return None

    def find_template(self, screenshot):
        """
        スクリーンショット内でテンプレート画像を探す
        
        Returns:
        --------
        tuple or None
            (中心x座標, 中心y座標, 類似度) または None（見つからない場合）
        """
        result = cv2.matchTemplate(screenshot, self.template_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val < self.threshold:
            print(f"テンプレートが見つかりませんでした（最大類似度: {max_val:.3f}）")
            return None

        # テンプレート画像の中心座標を計算
        h, w = self.template_image.shape[:2]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2

        return (center_x, center_y, max_val)

    def click_target(self, center_x, center_y):
        """指定された座標をクリック"""
        try:
            pyautogui.click(center_x, center_y)
            print(f"クリックを実行しました: ({center_x}, {center_y})")
            return True
        except Exception as e:
            print(f"エラー: クリックの実行に失敗しました: {str(e)}")
            return False

    def run(self):
        """メインの実行フロー"""
        # テンプレート画像の読み込み
        if not self.load_template():
            return False

        # スクリーンショットの取得
        screenshot = self.take_screenshot()
        if screenshot is None:
            return False

        # テンプレートの検索
        result = self.find_template(screenshot)
        if result is None:
            return False

        # 発見した位置をクリック
        center_x, center_y, similarity = result
        print(f"テンプレートを発見しました - 座標: ({center_x}, {center_y}), 類似度: {similarity:.3f}")
        return self.click_target(center_x, center_y)

def main():
    # テスト用のテンプレート画像パス（実際の画像に置き換える必要があります）
    template_path = "template.png"  # この部分は実際の画像パスに変更する必要があります
    
    # クリッカーのインスタンスを作成して実行
    clicker = TemplateMatchingClicker(template_path)
    clicker.run()

if __name__ == "__main__":
    main()