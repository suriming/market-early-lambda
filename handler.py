import json
import inference
import torch
import logging

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
infer = inference.Inference('model/A030210.pth', DEVICE)

def predict(event, context):
    try:
        # body = json.loads(event)
        context.log(event)
        input_data = event['time_series']
        input_data = [[item['open'], item['high'], item['low'], item['close'], item['vol'], item['value'],
                   item['agg_price'], item['foreign_rate'], item['agency_buy'], item['agency_netbuy']]
                  for item in input_data]
        input_data = torch.tensor(input_data, dtype=torch.float32)
        shape = input_data.shape
        input_data = input_data.unsqueeze(1)
        # input_data = input_data.permute(0, 2, 1) 
        preds = infer.autoencoder_anomaly(input_data=input_data)
        preds = preds.detach().tolist()
        # context.log(preds)
        # logging.info(f"prediction: {preds}")

        return {
            "statusCode": 200,
            "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
            "body": json.dumps({"prediction": preds})
        }
    except Exception as e:
        logging.error(e)
        return {
            "statusCode": 500,
            "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
            "body": json.dumps({"error": repr(e), "input_data": input_data.tolist()})
        }
    
    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
