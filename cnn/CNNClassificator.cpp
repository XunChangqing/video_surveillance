#include "CNNClassificator.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

CNNClassificator::CNNClassificator(QString configfilename, QString modelname, QString listname, QObject *parent)
    :QThread(parent)
{
    m_pCaffeNet = new Net<float>(configfilename.toAscii().data());
    m_pCaffeNet->CopyTrainedLayersFrom(modelname.toAscii().data());
    m_iCurTargetID = -1;
    m_iType = -1;
    Caffe::set_mode(Caffe::CPU);

    QFile listfile(listname);
    listfile.open(QIODevice::ReadOnly | QIODevice::Text);
    QTextStream in(&listfile);
    qDebug()<<"TypeNameList:";
    while (!in.atEnd()) {
        QString line = in.readLine();
        qDebug()<<line;
        m_TypeNameList.append(line);
    }
}

CNNClassificator::~CNNClassificator()
{
    if(m_pCaffeNet)
        delete m_pCaffeNet;
}

static bool ReadtoDatum(cv::Mat cv_img_origin, const int label,const int height, const int width, Datum* datum) {
    cv::Mat cv_img;
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
    if (!cv_img.data) {
        qDebug()<< "Could not Read img_data ";
        return false;
    }
    datum->set_channels(3);
    datum->set_height(cv_img.rows);
    datum->set_width(cv_img.cols);
    datum->set_label(label);
    datum->clear_data();
    datum->clear_float_data();
    string* datum_string = datum->mutable_data();
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < cv_img.rows; ++h) {
            for (int w = 0; w < cv_img.cols; ++w) {
                datum_string->push_back(
                                        static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
            }
        }
    }
    return true;
}

void CNNClassificator::plSetInputBlob(cv::Mat inMat, int targetID)
{
    m_InputBlob = inMat;
    m_iCurTargetID = targetID;
    //QString idname = QString("ID%1").arg(m_iCurTargetID);
    //cv::imshow(idname.toStdString(), inMat);
}

void CNNClassificator::run()
{
    if(m_InputBlob.empty())
        return;
    qDebug()<<"Start to classificate ID: "<<m_iCurTargetID;

    //get datum
    Datum datum;
    if (!ReadtoDatum(m_InputBlob, 1, 227, 227, &datum)) LOG(ERROR) << "Read to Datum Failed!!!";
    //get the blob
    Blob<float>* blob = new Blob<float>(1, datum.channels(), datum.height(), datum.width());
    //get the blobproto
    BlobProto blob_proto;
    blob_proto.set_num(1);
    blob_proto.set_channels(datum.channels());
    blob_proto.set_height(datum.height());
    blob_proto.set_width(datum.width());
    const int data_size = datum.channels() * datum.height() * datum.width();
    int size_in_datum = std::max<int>(datum.data().size(),            datum.float_data_size());
    for (int i = 0; i < size_in_datum; ++i) {
        blob_proto.add_data(0.);
    }
    const string& data = datum.data();
    if (data.size() != 0) {
        for (int i = 0; i < size_in_datum; ++i) {
            blob_proto.set_data(i, blob_proto.data(i) + (uint8_t)data[i]);
        }
    }

    //set data into blob
    blob->FromProto(blob_proto);
    //fill the vector
    vector<Blob<float>*> bottom;
    bottom.push_back(blob);
    float type = 0.0;

    const vector<Blob<float>*>& result =  m_pCaffeNet->Forward(bottom, &type);

    float max = 0;
    int max_i = 0;
    for(int i=0;i<m_TypeNameList.size();i++){
        float value = result[0]->cpu_data()[i];
        if (max < value){
            max = value;
            max_i = i;
        }
    }
    qDebug()<<"Finish class of target id:"<<m_iCurTargetID<<"\t type:"<<max_i<<"\tratio"<<max<<"\t name:"<<m_TypeNameList.at(max_i);
    if(max>=0.8)
    {
        m_iType = max_i;
    }
}
