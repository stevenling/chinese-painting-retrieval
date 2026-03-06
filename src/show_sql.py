"""
用于查看和管理 SQLite 数据库 paint.db 中 image 表的内容。

提供的功能：
- search(): 按指定标签查询 image 表中的记录（当前示例为 label='human'）
- delete_image(): 清空 image 表中的所有记录（危险操作，默认在 main 中注释掉）
"""

import sqlite3


def search():
    """
    按固定标签（当前为 'human'）查询 image 表中对应记录的 imgPath 列，并打印结果列表。
    可根据需要修改 nowLable 以查看不同类别的图像路径。
    """
    conn = sqlite3.connect('paint.db')
    cursor = conn.cursor()
    nowLable = "human"
    cursor.execute('select imgPath from image  where label = ?', [nowLable])  # limit 4
    # cursor.execute('select * from image')
    values = cursor.fetchall()  # 使用featchall获得结果集（list）
    print(values)  # result:[('1', 'Michael')]
    cursor.close()
    conn.close()


def delete_image():
    """
    删除 image 表中的所有记录，并打印删除后的查询结果（通常为空列表）。
    注意：这是不可逆操作，调用前请务必确认。
    """
    conn = sqlite3.connect('paint.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM image')
    conn.commit()
    # 使用 fetch all 获得结果集（list）
    values = cursor.fetchall()
    # result:[('1', 'Michael')]
    print(values)
    cursor.close()
    conn.close()


def main():
    # delete_image()
    search()


if __name__ == '__main__':
    main()
