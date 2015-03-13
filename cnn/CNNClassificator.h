#ifndef CNNCLASSIFICATOR_H
#define CNNCLASSIFICATOR_H

#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;
using namespace std;

class CNNClassificator : public QThread
{
    Q_OBJECT
public:
    //static QString splGetTypeName()
    explicit CNNClassificator(QString configfilename, QString modelname, QString listname , QObject *parent = 0);
    ~CNNClassificator();

    void plSetInputBlob(cv::Mat inMat, int targetID);
    int plGetCurTargetID(){return m_iCurTargetID;}
    int plGetType(){return m_iType;}
    QString plGetTypeName(){return m_TypeNameList.at(m_iType);}
    QString plGetTypeName(int type){
        QString dummy("error");
        if(type>=0 && type<m_TypeNameList.size())
            return m_TypeNameList.at(type);
        else
            return dummy;
    }
    void plReset(){m_iType = 0; m_iCurTargetID = -1;}

protected:
    void run();

signals:

public slots:

private:
    Net<float>* m_pCaffeNet;
    cv::Mat m_InputBlob;
    QStringList m_TypeNameList;
    int m_iCurTargetID;
    int m_iType;
};

#endif // CNNCLASSIFICATOR_H
