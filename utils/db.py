import mysql.connector

class db:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.mydb = mysql.connector.connect(
                                              host="localhost",
                                              user="nevin",
                                              passwd="password",
                                              database="anomaly"
                                        )
        self.mycursor = self.mydb.cursor()
    def push(self, args):
        query = "INSERT INTO anomaly(timestamp, score) " \
                "VALUES(%s,%s)"
        try:
            self.mycursor.execute(query, args)
            self.mydb.commit()
            if(self.verbose):
                print("Data committed")
        except Exception as E:
            if(self.verbose):
                print("Error ", E)
            self.mydb.rollback()
            print("Data committed")
