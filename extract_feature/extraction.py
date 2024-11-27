import os
import pandas as pd
from androguard.core.bytecodes.apk import APK
from collections import defaultdict

def extract_permissions(apk_path):
    try:
        apk = APK(apk_path)
        permissions = apk.get_permissions()
        return permissions
    except Exception as e:
        print(f"Error extracting permissions from {apk_path}: {e}")
        return []

def extract_features(apk_path, all_permissions):
    try:
        apk = APK(apk_path)

        # Lấy danh sách các quyền
        permissions = apk.get_permissions()

        # Biểu diễn quyền dưới dạng mảng nhị phân
        permission_vector = [1 if perm in permissions else 0 for perm in all_permissions]

        # Lấy thông tin khác nếu có
        min_sdk_version = apk.get_min_sdk_version() or "N/A"
        target_sdk_version = apk.get_target_sdk_version() or "N/A"
        version_code = apk.get_androidversion_code() or "N/A"
        version_name = apk.get_androidversion_name() or "N/A"

        features = {
            "file_name": os.path.basename(apk_path),
            "min_sdk_version": min_sdk_version,
            "target_sdk_version": target_sdk_version,
            "version_code": version_code,
            "version_name": version_name,
        }
        features.update({f"permission_{i}": v for i, v in enumerate(permission_vector)})
        return features
    except Exception as e:
        print(f"Error processing {apk_path}: {e}")
        return None

def process_directory(data_dir):
    # Lấy danh sách tất cả các quyền xuất hiện trong tập dữ liệu
    all_permissions = set()
    labels = os.listdir(data_dir)
    apk_files = []
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for apk_file in os.listdir(label_dir):
                if apk_file.endswith(".apk"):
                    apk_path = os.path.join(label_dir, apk_file)
                    print(f"Scanning permissions in {apk_path}...")
                    permissions = extract_permissions(apk_path)
                    all_permissions.update(permissions)
                    apk_files.append((apk_path, label))

    # Chuyển danh sách quyền thành một danh sách cố định
    all_permissions = sorted(list(all_permissions))

    # Trích xuất đặc trưng từ từng APK
    data = []
    for apk_path, label in apk_files:
        print(f"Processing features for {apk_path}...")
        features = extract_features(apk_path, all_permissions)
        if features:
            features["label"] = label
            data.append(features)

    return data, all_permissions

def save_to_csv(data, all_permissions, output_file):
    # Ghi file CSV, các quyền được lưu dưới dạng cột permission_0, permission_1, ...
    df = pd.DataFrame(data)
    permission_columns = [f"permission_{i}" for i in range(len(all_permissions))]
    df.columns = ["file_name", "min_sdk_version", "target_sdk_version", "version_code", "version_name"] + permission_columns + ["label"]
    df.to_csv(output_file, index=False)
    print(f"Saved features to {output_file}")

if __name__ == "__main__":
    data_dir = "data"
    output_file = "./data/cicmaldroid_features.csv"

    data, all_permissions = process_directory(data_dir)
    if data:
        save_to_csv(data, all_permissions, output_file)
    else:
        print("No data to save!")
