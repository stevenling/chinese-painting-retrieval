import sqlite3

def search():
    """
    查看数据库所有记录
    """
    conn = sqlite3.connect('paint.db')
    cursor = conn.cursor()
    nowLable = "human"
    cursor.execute('select imgPath from image  where label = ?',[nowLable]) #limit 4
    #cursor.execute('select * from image')
    values = cursor.fetchall()     #使用featchall获得结果集（list）
    print(values) #result:[('1', 'Michael')]
    cursor.close()
    conn.close()

def delteImage():
    """
    删除数据库所有记录
    """
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