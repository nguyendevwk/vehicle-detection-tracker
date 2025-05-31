import os
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import threading


# Tạo thư mục lưu ảnh nếu chưa có
save_folder = "dataset"
os.makedirs(save_folder, exist_ok=True)
BASE_FOLDER ='dataset'
# Danh sách URL camera
links = [
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=662a8f3a1afb9c00172d2b31&camLocation=Nguy%E1%BB%85n%20H%E1%BB%AFu%20Th%E1%BB%8D%20-%20%C4%90%C6%B0%E1%BB%9Dng%20s%E1%BB%91%2015&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=662b4e8e1afb9c00172d865c&camLocation=L%C3%BD%20Th%C6%B0%E1%BB%9Dng%20Ki%E1%BB%87t%20-%20Nguy%E1%BB%85n%20Ch%C3%AD%20Thanh&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5a6060a88576340017d0660d&camLocation=Nguy%E1%BB%85n%20Th%C3%A1i%20S%C6%A1n%20-%20Phan%20V%C4%83n%20Tr%E1%BB%8B%20%201&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5a6060a88576340017d0660d&camLocation=Nguy%E1%BB%85n%20Th%C3%A1i%20S%C6%A1n%20-%20Phan%20V%C4%83n%20Tr%E1%BB%8B%20%201&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u80",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5deb576d1dc17d7c5515acf2&camLocation=%C4%90i%E1%BB%87n%20Bi%C3%AAn%20Ph%E1%BB%A7%20-%20C%C3%A1ch%20M%E1%BA%A1ng%20Th%C3%A1ng%20T%C3%A1m&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=662b58bd1afb9c00172d9119&camLocation=L%C3%AA%20V%C4%83n%20S%E1%BB%B9%20-%20%C4%90%E1%BA%B7ng%20V%C4%83n%20Ng%E1%BB%AF&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=662b5b8c1afb9c00172d92ca&camLocation=B%C3%ACnh%20Long%20-%20L%C3%AA%20Th%C3%BAc%20Ho%E1%BA%A1ch&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u84",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=6623f1046f998a001b2527db&camLocation=L%C5%A9y%20B%C3%A1n%20B%C3%ADch%20-%20Tho%E1%BA%A1i%20Ng%E1%BB%8Dc%20H%E1%BA%A7u&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=662b57471afb9c00172d9095&camLocation=C%E1%BB%99ng%20H%C3%B2a%20-%20%E1%BA%A4p%20B%E1%BA%AFc&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=6623f0576f998a001b2527ac&camLocation=%C3%82u%20C%C6%A1%20-%20Tho%E1%BA%A1i%20Ng%E1%BB%8Dc%20H%E1%BA%A7u&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
"https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=6623e7076f998a001b2523ea&camLocation=L%C3%BD%20Th%C3%A1i%20T%E1%BB%95%20-%20S%C6%B0%20V%E1%BA%A1n%20H%E1%BA%A1nh&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8",
]


# Thiết lập Chrome chạy không hiển thị giao diện (headless)
chrome_options = Options()
chrome_options.add_argument("--headless")  # Chạy không có giao diện
chrome_options.add_argument("--disable-gpu")  # Tắt GPU tăng hiệu suất
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Mở trình duyệt Chrome với Selenium
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)



# Hàm tải ảnh từ URL
def download_image(img_url, save_path):
    """Tải ảnh từ URL và lưu vào thư mục dataset"""
    try:
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"[✅] Đã lưu: {save_path}")
        else:
            print(f"[❌] Không tải được ảnh từ: {img_url}")
    except Exception as e:
        print(f"[⚠] Lỗi tải ảnh {img_url}: {e}")

# Hàm chính để tải ảnh mỗi 30 giây
def fetch_images():
    while True:
        current_date = datetime.now().strftime("%Y-%m-%d")
        save_folder = os.path.join(BASE_FOLDER, current_date)

        # Tạo thư mục theo ngày nếu chưa có
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for idx, url in enumerate(links):
            print(f"\n[🚀] Đang tải dữ liệu từ: {url}")

            # Truy cập trang web
            driver.get(url)
            driver.implicitly_wait(5)  # Chờ trang tải

            # Lấy HTML sau khi trang đã load xong
            html_after_load = driver.page_source
            soup = BeautifulSoup(html_after_load, "html.parser")

            # Tìm tất cả thẻ <img>
            img_tags = soup.find_all("img")

            # Kiểm tra và tải ảnh
            for i, img in enumerate(img_tags):
                img_url = img.get("src")
                if img_url:
                    if not img_url.startswith("http"):  # Nếu URL không đầy đủ, thêm domain
                        img_url = "https://giaothong.hochiminhcity.gov.vn" + img_url

                    # Đặt tên file theo timestamp + index
                    timestamp = datetime.now().strftime("%H%M%S")
                    img_name = f"camera_{idx+1}_{i+1}_{timestamp}.jpg"
                    save_path = os.path.join(save_folder, img_name)

                    # Tải ảnh
                    download_image(img_url, save_path)
                    time.sleep(1)  # Tránh tải quá nhanh gây lỗi

        print("\n✅ Tất cả ảnh đã được tải xong! Chờ 30 giây...\n")
        time.sleep(10)  # Đợi 30 giây trước khi tải tiếp

# Chạy vòng lặp tải ảnh trong luồng riêng (không làm treo chương trình chính)
thread = threading.Thread(target=fetch_images, daemon=True)
thread.start()

# Chương trình chạy mãi (có thể thêm giao diện Tkinter nếu cần)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[⚠] Dừng chương trình!")
    driver.quit()