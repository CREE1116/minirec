import os
import csv
import sys

def detect_separator(file_path, num_lines=5):
    """
    파일의 구분자(separator)를 자동으로 탐지합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    # 후보 구분자 리스트
    possible_separators = [',', '\t', ';', '|', ' ']
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = ""
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break
                sample += line

        if not sample:
            print("Error: File is empty.")
            return None

        # csv.Sniffer를 사용하여 탐지 시도
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=possible_separators)
            return dialect.delimiter
        except:
            # Sniffer가 실패할 경우 수동 카운팅 (가장 많이 등장하는 문자 선택)
            counts = {sep: sample.count(sep) for sep in possible_separators}
            best_sep = max(counts, key=counts.get)
            if counts[best_sep] > 0:
                return best_sep
            return None

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/detect_separator.py <file_path>")
        sys.exit(1)

    target_file = sys.argv[1]
    separator = detect_separator(target_file)

    if separator:
        readable_sep = separator.replace('\t', '\\t').replace(' ', 'space')
        print(f"Detected Separator: '{readable_sep}'")
    else:
        print("Could not detect a valid separator.")
