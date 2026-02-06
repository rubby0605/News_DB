import os
from bs4 import BeautifulSoup

# 創建輸出目錄
output_dir = 'extracted_html'
os.makedirs(output_dir, exist_ok=True)

# 讀取 .webarchive 文件
with open('homepage.webarchive', 'rb') as file:
    raw_data = file.read()

# 嘗試使用不同編碼解碼
try:
    content = raw_data.decode('utf-8')
except UnicodeDecodeError:
    content = raw_data.decode('latin1')

# 解析 HTML
soup = BeautifulSoup(content, 'lxml')

# 假設所有 HTML 文件都在 <html> 標籤下，並提取
for idx, html_content in enumerate(soup.find_all('html')):
    # 保存每個 HTML 文件
    filename = os.path.join(output_dir, f'page_{idx + 1}.html')
    with open(filename, 'w', encoding='utf-8') as output_file:
        output_file.write(html_content.prettify())

print(f"Extracted {len(soup.find_all('html'))} HTML files to '{output_dir}'")
