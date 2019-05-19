import sqlite3
# 创建数据库
def creatDb():
    conn = sqlite3.connect('paint.db')
    cursor = conn.cursor()
    cursor.execute('create table image(id integer primary key, label varchar(30), imgPath varchar(100), feature BLOB)')
    #cursor.execute('insert into image (id, label, imgPath, feature) values (1, \'flowerbird\', \'paint_photos\flowerBird\', NULL)')
    #通过rowcount获得插入的行数：
    print(cursor.rowcount) #reusult 1
    cursor.close()
    conn.commit()
    conn.close()

def main():
    creatDb()
if __name__ == '__main__':
    main()