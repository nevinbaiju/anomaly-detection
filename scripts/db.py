from mysql import connector

class db:
    def __init__(self):
        mydb = mysql.connector.connect(
                                          host="localhost",
                                          user="arundhati",
                                          passwd="arundhati",
                                          database="user"
                                        )
        self.mycursor = mydb.cursor()
    def push(self, args):
        query = "INSERT INTO anomaly(timestamp, score) " \
                "VALUES(%s,%s)"
        try:
            self.mycursor.execute(query, args)
            mydb.commit()
            if(verbose):
                print("Data committed")
        except:
            if(verbose):
                print("Error")
            mydb.rollback()
            print("Data committed")
