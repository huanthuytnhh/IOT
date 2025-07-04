import sqlite3
import numpy as np
import json

from models.appliance import Appliance
from models.member import Member
from models.permission import Permission

def connect_db(db_path):
    """ Tạo kết nối đến cơ sở dữ liệu SQLite được chỉ định bởi db_path """
    conn = sqlite3.connect(db_path)
    return conn

def query_members(conn):
    cursor = conn.cursor()

    # Truy vấn dữ liệu từ bảng Member và tạo các đối tượng Member
    cursor.execute("SELECT * FROM members_member")
    members = [Member(*row) for row in cursor.fetchall()]
    return members

def query_appliances(conn):
    cursor = conn.cursor()

    # Truy vấn dữ liệu từ bảng Appliance và tạo các đối tượng Appliance
    cursor.execute("SELECT * FROM appliances_appliance")
    appliances = [Appliance(*row) for row in cursor.fetchall()]
    return appliances

def query_permissions(conn):
    cursor = conn.cursor()

    # Truy vấn liên kết ba bảng để lấy thông tin cần thiết
    cursor.execute("""
    SELECT p.id, p.member_id, p.appliance_id, m.name as member_name, a.name as appliance_name
    FROM permissions_permission p
    JOIN members_member m ON p.member_id = m.id
    JOIN appliances_appliance a ON p.appliance_id = a.id
    """)
    
    permissions = [Permission(*row) for row in cursor.fetchall()]
    return permissions

def query_members_files(conn):
    cursor = conn.cursor()
    cursor.execute("""
    SELECT mf.member_id, mf.file, mf.features, mm.name
    FROM members_memberfile mf
    JOIN members_member mm ON mf.member_id = mm.id
    """)
    member_files = [{
        'member_id': row[0],
        'file_path': row[1],
        'features': row[2],
        'member_name': row[3]
    } for row in cursor.fetchall()]
    return member_files

def query_member_files(conn, member_name):
    cursor = conn.cursor()
    cursor.execute("""
    SELECT mf.member_id, mf.file, mf.features, mm.name
    FROM members_memberfile mf
    JOIN members_member mm ON mf.member_id = mm.id
    WHERE mm.name = ?
    """, (member_name,))
    member_files = [{
        'member_id': row[0],
        'file_path': row[1],
        'features': row[2],
        'member_name': row[3]
    } for row in cursor.fetchall()]
    return member_files

def print_tables_in_database(db_path):
    # Kết nối đến cơ sở dữ liệu SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Truy vấn để lấy tên của tất cả các bảng trong cơ sở dữ liệu
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # In ra tên của các bảng
    print("Các bảng có trong cơ sở dữ liệu:")
    for table in tables:
        print(table[0])

    # Đóng kết nối đến cơ sở dữ liệu
    conn.close()
    
def get_features(features):
    """Chuyển chuỗi JSON trở lại thành list np.ndarray."""
    if features:
        # Đọc chuỗi JSON và chuyển nó thành một list
        features_list = json.loads(features)
        # Chuyển list thành list of np.ndarray
        return [np.array(feature) for feature in features_list]
    return None
# conn = connect_db("/home/tranductri2003/Code/PBL05_smart_home_with_voice_print_and_antifraud_ai/BackEnd/db.sqlite3")
# appliances = query_appliances(conn)
# permissions = query_permissions(conn)
# members = query_members(conn)

# print("Danh sách các thành viên trong nhà:")
# for member in members:
#     print(member.name)

# print("Danh sách các thiết bị trong nhà:")
# for appliance in appliances:
#     print(appliance.name)

# print("Danh sách các quyền điều khiển:")
# for permission in permissions:
#     print(f"{permission.member_name} có quyền điều khiển {permission.appliance_name}")
    
# member_files = query_members_files(conn)
# print("Danh sách các vector của thành viên:")
# for file in member_files:
#     print(f"Member: {file['member_name']} - File: {file['file_path']} - Features: {file['features']}")
    
# # Gọi hàm để in ra tất cả các bảng trong cơ sở dữ liệu
# # print_tables_in_database("D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\BackEnd\db.sqlite3")

# print("Các vector của Trần Đức Trí")
# tri_vectors = query_member_files(conn, "Trần Đức Trí")
# for vector in tri_vectors:
#     print(vector['features'], len(vector['features']), len(get_features(vector['features'])))
#     print()
#     print()
#     print()