import sqlite3

# 查看数据库所有记录
def search():
    conn = sqlite3.connect('paint.db')
    cursor = conn.cursor()
    nowLable = "human"
    cursor.execute('select imgPath from image  where label = ?',[nowLable]) #limit 4
    #cursor.execute('select * from image')
    values = cursor.fetchall()     #使用featchall获得结果集（list）
    print(values) #result:[('1', 'Michael')]
    cursor.close()
    conn.close()

# 删除数据库所有记录
def delteImage():
    conn = sqlite3.connect('paint.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM image')
    conn.commit()
    values = cursor.fetchall() #使用featchall获得结果集（list）
    print(values) #result:[('1', 'Michael')]
    cursor.close()
    conn.close()
def main():
    #delteImage()
    search()
if __name__ == '__main__':
    main()