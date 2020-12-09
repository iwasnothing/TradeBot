gcloud pubsub subscriptions create stock_pair_handler --topic stock_pair \
   --push-endpoint=https://helloworld-s3yk5iivva-uc.a.run.app/correlation \
   --push-auth-service-account=pubsubsa@iwasnothing-self-learning.iam.gserviceaccount.com

gcloud pubsub subscriptions create stock_filter_handler --topic stock_filter \
   --push-endpoint=https://helloworld-s3yk5iivva-uc.a.run.app/filter \
   --push-auth-service-account=pubsubsa@iwasnothing-self-learning.iam.gserviceaccount.com

gcloud pubsub subscriptions create stock_train_handler --topic stock_train \
   --push-endpoint=https://helloworld-s3yk5iivva-uc.a.run.app/train \
   --push-auth-service-account=pubsubsa@iwasnothing-self-learning.iam.gserviceaccount.com

gcloud pubsub subscriptions create stock_predict_handler --topic stock_predict \
   --push-endpoint=https://helloworld-s3yk5iivva-uc.a.run.app/predict \
   --push-auth-service-account=pubsubsa@iwasnothing-self-learning.iam.gserviceaccount.com

