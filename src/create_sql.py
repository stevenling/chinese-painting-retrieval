"""
创建 SQLite 数据库和 image 表。

Author: yunhu
Date: 2019/5/19
"""
import sqlite3

import config


def create_db():
    """
    创建数据库
    """
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute('create table image(id integer primary key, label varchar(30), imgPath varchar(100), feature BLOB)')
    # cursor.execute('insert into image (id, label, imgPath, feature) values ...')
    # 通过 rowcount 获得插入的行数
    print(cursor.rowcount)
    cursor.close()
    conn.commit()
    conn.close()


def main():
    create_db()

if __name__ == '__main__':
    main()