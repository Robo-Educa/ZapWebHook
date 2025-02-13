# Carrega bibliotecas 
from flask import Flask, request, jsonify
import os
import requests
import time
import google.generativeai as genai
from google.cloud import storage 
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
load_dotenv()  

#region - Variáveis de ambiente
my_api_key = os.environ.get("API_KEY")                        # Gemini - API_KEY
system_instruction = os.environ.get("SYSTEM_INSTRUCTIONS")    # Gemini - Instruções do Sistema / Informa as caracteristicas do Assistente.
url_base = os.environ.get("URL_BASE")                         # WhatsApp Cloud API - URL base da API (incluindo versão)
token = os.environ.get("TOKEN")                               # WhatsApp Cloud API - Token de segurança para acesso às mensagens 
audio_bucket_name = os.environ.get("AUDIO_BUCKET_NAME")       # Google Cloud Storage - Nome do Bucket para armazenamento de AUDIOS
image_bucket_name = os.environ.get("IMAGE_BUCKET_NAME")       # Google Cloud Storage - Nome do Bucket para armazenamento de IMAGENS
path_credential = os.environ.get("PATH_FB_CREDENTIAL")        # Path para JSON contendo Google Firebase Admin SDK - Credencial acesso
path_audio_messages = os.environ.get("PATH_AUDIO_MESSAGES")   # Path para armazenamento de arquivos de audio
path_image_messages = os.environ.get("PATH_IMAGE_MESSAGES")   # Path para armazenamento de imagens
#endregion

#region - Inicia modelos Google Generative AI
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}
audio_generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
}
image_generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]
genai.configure(api_key=my_api_key)
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest",
        generation_config=generation_config,
        system_instruction=system_instruction,
        safety_settings=safety_settings)
audio_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest",
        generation_config=audio_generation_config,
        safety_settings=safety_settings)
image_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest",
        generation_config=image_generation_config,
        safety_settings=safety_settings)
#endregion

#region - Inicializa o Firebase app (Gestão de Banco No-SQL ref. histórico de mensagens)
cred = credentials.Certificate(path_credential)             # Volume criado dentro do container, na console do Google Cloud Run
firebase_admin.initialize_app(cred)
db = firestore.client()
#endregion

app = Flask(__name__)

# Endpoint POST para recebimento de notificações da WhatsApp Cloud API
@app.route("/webhook", methods=["POST"])
def webhook():   
    data = request.json    
    handled = False

    # Verifica destinatário.
    if data.get("entry") and data["entry"][0].get("changes"):
        change = data["entry"][0]["changes"][0]
        if change.get("value") and change["value"].get("metadata"):            
            
            metadata = change["value"]["metadata"]
            phone_number_id = metadata.get("phone_number_id")
            id_tel = os.environ.get("ID_TEL") 

            if phone_number_id != id_tel:
                return jsonify({"status": "Ok"}), 200

    # Tratamento de mensagens recebidas
    if data.get("entry") and data["entry"][0].get("changes"):
        change = data["entry"][0]["changes"][0]
        if change.get("value") and change["value"].get("messages"):
            
            message = change["value"]["messages"][0]
            tel = message.get("from")
            type_message = message.get("type")
            id_text = message.get("id")

            # Verifica se contato já existe na base de dados
            contact = exist_contact(tel)
            if contact == False:
                store_contact(tel)
                message_history = []
            else:
                message_history = get_menssages(tel)                # Obtem histórico de mensagens

            # Tratamento de Mensagens de TEXTO
            if type_message == "text":
                if exist_idText(id_text):                           # Validação para evitar duplicidade de lançamentos, caso a WhatsApp Cloud API envie a mesma mensagem repetidamente
                    return 
                
                body_message = message.get("text").get("body")      # Texto da mensagem digitada pelo usuário      
                role = "user"                                       # role=user => mensagem enviada pelo usuário
                store_message(tel, role, body_message)              # Salva mensagem em banco NO-SQL. 
                store_idText(id_text)                               # Salva ID do texto para posterior validação de duplicidade
                handled = True

                if body_message.upper() == "PARAR MENSAGENS":
                    contact_update_status(tel, "Inativos")
                    treated_response = "Ok, *não iremos lhe enviar novas mensagens*. Caso tenha solicitado por engano, digite a palavra *ATIVAR CADASTRO*."
                    send_message = send_text_message(tel, treated_response)     # Envia resposta de volta para o usuário através da WhatsApp Cloud API                
                    if send_message:
                        role = "model"                                          # role=model => mensagem enviada pela IA
                        store_message(tel, role, treated_response)              # Salva mensagem em banco NO-SQL. 
                    return
                elif body_message.upper() == "ATIVAR CADASTRO":
                    contact_update_status(tel, "Ativos")
                    treated_response = "Ok, *cadastro ATIVADO*."
                    send_message = send_text_message(tel, treated_response)     # Envia resposta de volta para o usuário através da WhatsApp Cloud API                
                    if send_message:
                        role = "model"                                          # role=model => mensagem enviada pela IA
                        store_message(tel, role, treated_response)              # Salva mensagem em banco NO-SQL. 
                    return

                convo = model.start_chat(history = message_history) # Inicia chat, contextualizando a IA com o histórico da conversação
                convo.send_message(body_message)                    # envia nova mensagem para ser processada pela IA
                response = convo.last.text                          # Obtem resposta da IA

                treated_response, instruction = response_treatment(response)    # Verifica se existem instruções ou comandos enviados pela IA e faz a devida separação da mensagem

                send_message = send_text_message(tel, treated_response)     # Envia resposta de volta para o usuário através da WhatsApp Cloud API                
                if send_message:
                    role = "model"                                          # role=model => mensagem enviada pela IA
                    store_message(tel, role, treated_response)              # Salva mensagem em banco NO-SQL. 
                
                if instruction != "":                                       # Caso exista alguma instrução, analisa a mesma e dá o tratamento devido
                    handle_instruction(instruction, tel)
            
            # Tratamento de Mensagens de AUDIO
            elif type_message == "audio":                
                id_media = message.get("audio").get("id")   
                if exist_idMedia(id_media):                         # Validação para evitar duplicidade de lançamentos, caso a WhatsApp Cloud API envie a mesma mensagem repetidamente
                    return                 
                url_media, mime_type = get_url_media(id_media)                 # obtem URL do audio (Midia protegida por token - WhastApp Cloud API)
                if url_media:               
                    media = download_media(url_media, tel)          # faz o download do audio em formato binário 
                    if media:
                        handled = True
                        file_name = store_audio(media, tel, mime_type)         # Salva audio em bucket do Google Cloud Storage 
                        if file_name:
                            store_idMedia(id_media)
                            try:    
                                # Realiza a transcrição de Audio para Texto (Speech-to-Text) utilizando Google Gemini / multimodal prompt
                                path_media = f"/{path_audio_messages}/{file_name}"                          
                                audio_media = genai.upload_file(path=path_media, mime_type=mime_type)
                                audio_analysis = audio_model.generate_content(["Transcreva este audio", audio_media])  
                                audio_transcript = audio_analysis.text  

                                role = "user"                                       # role=user => mensagem enviada pelo usuário
                                store_message(tel, role, audio_transcript)          # Salva mensagem em banco NO-SQL.                                            

                                convo = model.start_chat(history = message_history) # Inicia chat, contextualizando a IA com o histórico da conversação                                
                                convo.send_message(audio_transcript)                # envia mensagem recebida para ser processada pela IA
                                response = convo.last.text                          # Obtem resposta da IA

                                treated_response, instruction = response_treatment(response)    # Verifica se existem instruções ou comandos enviados pela IA e faz a devida separação da mensagem
                                
                                send_message = send_text_message(tel, treated_response)         # Envia resposta de volta para o usuário através da WhatsApp Cloud API                
                                if send_message:
                                    role = "model"                                              # role=model => mensagem enviada pela IA
                                    store_message(tel, role, treated_response)                  # Salva mensagem em banco NO-SQL. 

                                if instruction != "":                                           # Caso exista alguma instrução, analisa a mesma e dá o tratamento devido
                                    handle_instruction(instruction, tel)
                            except Exception as e:
                                response = "Opa, algo deu errado e não consegui analisar sua mensagem. Tente novamente"
                                send_message = send_text_message(tel, response)
                                if send_message:
                                    role = "model"                                  # role=model => mensagem enviada pela IA
                                    store_message(tel, role, response)              # Salva mensagem em banco NO-SQL. 
                                insert_internal_error("audio_analysis", f"Exception - {e}", tel)                          
                        else:
                            send_text_message(tel, "Não foi possível salvar o Audio na Nuvem. Tente Novamente") 
                            insert_internal_error("audio_store", "Não foi possível salvar o Audio na Nuvem.", tel)
                    else:
                        send_text_message(tel, "Não foi possível obter o Audio. Tente Novamente") 
                        insert_internal_error("audio_get", "Não foi possível obter o Audio junto a WhatsApp Cloud API", tel)
                else:
                    send_text_message(tel, "Não foi possível obter a URL do Audio. Tente Novamente") 
                    insert_internal_error("audio_get", "Não foi possível obter a URL do Audio junto a WhatsApp Cloud API", tel)

            # Tratamento de Mensagens com Imagem
            elif type_message == "image":                
                id_media = message.get("image").get("id")   
                if exist_idMedia(id_media):                         # Validação para evitar duplicidade de lançamentos, caso a WhatsApp Cloud API envie a mesma mensagem repetidamente
                    return                 
                url_media, mime_type = get_url_media(id_media)      # obtem URL da Imagem (Midia protegida por token - WhastApp Cloud API)
                if url_media:               
                    media = download_media(url_media, tel)          # faz o download da imagem em formato binário 
                    if media:
                        handled = True
                        file_name = store_image(media, tel, mime_type)              # Salva imagem em bucket do Google Cloud Storage 
                        if file_name:
                            store_idMedia(id_media)
                            try:    
                                path_media = f"/{path_image_messages}/{file_name}"                          
                                role = "user"                                       # role=user => mensagem enviada pelo usuário
                                describe = f"Imagem recebida. Path: {path_media}"
                                store_message(tel, role, describe)                  # Salva mensagem com descrição da imagem em banco NO-SQL.   
                                update_contact_last_media(tel, file_name)           # Atualiza Contato com File_Name da ultima imagem recebida                                   

                                convo = model.start_chat(history = message_history) # Inicia chat, contextualizando a IA com o histórico da conversação                                
                                convo.send_message(describe)                        # envia mensagem recebida para ser processada pela IA
                                response = convo.last.text                          # Obtem resposta da IA

                                treated_response, instruction = response_treatment(response)    # Verifica se existem instruções ou comandos enviados pela IA e faz a devida separação da mensagem
                                
                                send_message = send_text_message(tel, treated_response)         # Envia resposta de volta para o usuário através da WhatsApp Cloud API                
                                if send_message:
                                    role = "model"                                              # role=model => mensagem enviada pela IA
                                    store_message(tel, role, treated_response)                  # Salva mensagem em banco NO-SQL. 

                                if instruction != "":                                           # Caso exista alguma instrução, analisa a mesma e dá o tratamento devido
                                    handle_instruction(instruction, tel)
                            except Exception as e:
                                response = "Opa, algo deu errado e não consegui analisar sua imagem. Tente novamente"
                                send_message = send_text_message(tel, response)
                                if send_message:
                                    role = "model"                                  # role=model => mensagem enviada pela IA
                                    store_message(tel, role, response)              # Salva mensagem em banco NO-SQL. 
                                insert_internal_error("image_analysis", f"Exception - {e}", tel)                          
                        else:
                            send_text_message(tel, "Não foi possível salvar a Imagem na Nuvem. Tente Novamente") 
                            insert_internal_error("image_store", "Não foi possível salvar a Imagem na Nuvem.", tel)
                    else:
                        send_text_message(tel, "Não foi possível obter a Imagem. Tente Novamente") 
                        insert_internal_error("image_get", "Não foi possível obter a Imagem junto a WhatsApp Cloud API", tel)
                else:
                    send_text_message(tel, "Não foi possível obter a URL da Imagem. Tente Novamente") 
                    insert_internal_error("image_get", "Não foi possível obter a URL da Imagem junto a WhatsApp Cloud API", tel)
            
            elif type_message == "button":
                if exist_idText(id_text):                           # Validação para evitar duplicidade de lançamentos, caso a WhatsApp Cloud API envie a mesma mensagem repetidamente
                    return 
                
                body_message = message.get("button").get("text")    # Texto do botão
                role = "user"                                       # role=user => mensagem enviada pelo usuário
                store_message(tel, role, body_message)              # Salva mensagem em banco NO-SQL. 
                store_idText(id_text)                               # Salva ID do texto para posterior validação de duplicidade
                handled = True

                if body_message.upper() == "PARAR MENSAGENS":
                    contact_update_status(tel, "Inativos")
                    treated_response = "Ok, *não iremos lhe enviar novas mensagens*. Caso tenha solicitado por engano, digite a palavra *ATIVAR CADASTRO*."
                    send_message = send_text_message(tel, treated_response)     # Envia resposta de volta para o usuário através da WhatsApp Cloud API                
                    if send_message:
                        role = "model"                                          # role=model => mensagem enviada pela IA
                        store_message(tel, role, treated_response)              # Salva mensagem em banco NO-SQL. 
                    return
                elif body_message.upper() == "ATIVAR CADASTRO":
                    contact_update_status(tel, "Ativos")
                    treated_response = "Ok, *cadastro ATIVADO*."
                    send_message = send_text_message(tel, treated_response)     # Envia resposta de volta para o usuário através da WhatsApp Cloud API                
                    if send_message:
                        role = "model"                                          # role=model => mensagem enviada pela IA
                        store_message(tel, role, treated_response)              # Salva mensagem em banco NO-SQL. 
                    return
                else:
                    contact_update_status(tel, "Ativos")            # Altera status do contato de NOVOS para ATIVOS

                convo = model.start_chat(history = message_history) # Inicia chat, contextualizando a IA com o histórico da conversação
                convo.send_message(body_message)                    # envia nova mensagem para ser processada pela IA
                response = convo.last.text                          # Obtem resposta da IA

                treated_response, instruction = response_treatment(response)    # Verifica se existem instruções ou comandos enviados pela IA e faz a devida separação da mensagem

                send_message = send_text_message(tel, treated_response)     # Envia resposta de volta para o usuário através da WhatsApp Cloud API                
                if send_message:
                    role = "model"                                          # role=model => mensagem enviada pela IA
                    store_message(tel, role, treated_response)              # Salva mensagem em banco NO-SQL. 
                
                if instruction != "":                                       # Caso exista alguma instrução, analisa a mesma e dá o tratamento devido
                    handle_instruction(instruction, tel)

            elif type_message == "reaction":
                handled = True
                # ... my code => trata Reactions
                return
            
            # Outros tipos de mensagens (imagens, figurinhas, localização, contato, etc)                                             
            else:               
                resposta = f"Desculpe ainda não fui programado para analisar mensagens do tipo: *{type_message}*. Envie somente Texto, Áudio ou Imagens"
                send_text_message(tel, resposta)    # envia resposta de volta para o usuário através da WhatsApp Cloud API       

    # Tratamento de ACK (status da mensagem: aceita, enviada, entregue,lida)
    if data.get("entry") and data["entry"][0].get("changes"):
        change = data["entry"][0]["changes"][0]
        if change.get("value") and change["value"].get("statuses"):
            statuses = data["entry"][0]["changes"][0]["value"]["statuses"][0]
            if statuses.get("id"):
                id_message = data["entry"][0]["changes"][0]["value"]["statuses"][0]["id"]
                timestamp = data["entry"][0]["changes"][0]["value"]["statuses"][0]["timestamp"]
                status = data["entry"][0]["changes"][0]["value"]["statuses"][0]["status"]        
                handled = True
                # ... my code => trata ACK   
    
    # Atualização de Status de Modelos 
    if data.get("entry") and data["entry"][0].get("changes"):
        change = data["entry"][0]["changes"][0]
        if change.get("field") == "message_template_status_update":
            event = data["entry"][0]["changes"][0]["value"]["event"] 
            message_template_id = data["entry"][0]["changes"][0]["value"]["message_template_id"]
            reason = data["entry"][0]["changes"][0]["value"]["reason"]

            other_info = ""
            if "other_info" in data["entry"][0]["changes"][0]["value"]:
                other_info_dict = data["entry"][0]["changes"][0]["value"]["other_info"]
                if other_info_dict:  
                    title = other_info_dict.get("title", "")  
                    description = other_info_dict.get("description", "")  
                    other_info = f"{title} - {description}"  

            handled = True
            update_campaign_status(event, message_template_id, reason)                          # Atualiza Status da Campanha
            
            if other_info != "": 
                campaign_name = get_campaign_name(message_template_id)
                alert = f"Mudança de STATUS da campanha {campaign_name}. Novo status = {other_info}"         # Salva Alerta
                store_campaign_alert(alert)

    # Atualização de Score de Qualidade de Modelos 
    if data.get("entry") and data["entry"][0].get("changes"):
        change = data["entry"][0]["changes"][0]
        if change.get("field") == "message_template_quality_update":
            previus = data["entry"][0]["changes"][0]["value"]["previous_quality_score"] 
            new = data["entry"][0]["changes"][0]["value"]["new_quality_score"]   
            message_template_id = data["entry"][0]["changes"][0]["value"]["message_template_id"]

            handled = True
            update_campaign_score(message_template_id, previus, new)            # Atualiza Score da Campanha
            
            campaign_name = get_campaign_name(message_template_id)
            alert = f"Mudança de SCORE da campanha {campaign_name}. Novo score = {new}"     # Salva Alerta
            store_campaign_alert(alert)

    # Salva notificação não analisada
    if handled == False:    
        untreated_notification(data)    # Salva notificação não analisada- Atenção!!! Somente em ambiente de Desenvolvimento para eventuais análises. Desabilitar em ambiente de Produção    
    
    return jsonify({"status": "Ok"}), 200

# Endpoint GET para validação do webhook junto a WhatsApp Cloud API
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    verify_token = os.environ.get("VERIFY_TOKEN")
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == verify_token:
            return challenge, 200
        else:
            return "Verification failed", 403
    else:
        return "Invalid request", 400

# Envia mensagem de texto para a WhatsApp Cloud API
def send_text_message(tel, text_response):
    
    url_base = os.environ.get("URL_BASE") 
    id_tel = os.environ.get("ID_TEL") 
    token = os.environ.get("TOKEN") 

    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": tel,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": text_response
        }
    }

    url = f"{url_base}/{id_tel}/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        json_response = response.json()
        if json_response.get("messages") and json_response["messages"][0].get("id"):            
            return True  # Indica sucesso
    else:
        return False  # Indica falha        

# Obtem URL do audio enviado pela WhatsApp Cloud API
def get_url_media(id_media):    
    url = f"{url_base}/{id_media}"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        url_response = response.json().get("url")
        mime_type = response.json().get("mime_type")
        return url_response, mime_type
    else:
        return False, False

# Realiza o Download de midia (audio/video) 
def download_media(url_media, tel):    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.get(url_media, headers=headers, stream=True)
        response.raise_for_status()  
        return response.content
    except requests.exceptions.RequestException as e:
        error_message = f"Erro ao baixar o arquivo de áudio: {e}"
        insert_internal_error("download_media", error_message, tel)
        return False
    except Exception as e:
        error_message = f"Erro ao salvar no Cloud Storage: {e}"
        insert_internal_error("download_media", error_message, tel)
        return False
    
# Salva Audio em Bucket do Google Cloud Storage e retorna seu nome
def store_audio(media, tel, mime_type):
    storage_client = storage.Client()           
    bucket = storage_client.bucket(audio_bucket_name)         # Bucket em que será salvo a midia

    if mime_type == "audio/aac":
        extension = "aac"
    elif mime_type == "audio/amr":
        extension = "amr"
    elif mime_type == "audio/mpeg":
        extension = "mp3"
    elif mime_type == "audio/mp4":
        extension = "m4a"        
    elif mime_type == "audio/ogg":
        extension = "ogg"        

    file_name = f"{tel}_{int(time.time())}.{extension}"         # Nome do arquivo a ser salvo (concatena número do telefone + timestamp)
    blob = bucket.blob(file_name)               
    try:
        blob.upload_from_string(media, content_type=mime_type)  # Realiza o Upload do arquivo para o Cloud Storage      
        return file_name        
    except Exception as e:
        error_message = f"Erro ao tentar salvar Áudio no Bucket da Google Cloud Storage. Detalhes: {e}"
        insert_internal_error("store_audio", error_message, tel)
        return False

# Salva Imagem em Bucket do Google Cloud Storage e retorna seu nome
def store_image(media, tel, mime_type):
    storage_client = storage.Client()           
    bucket = storage_client.bucket(image_bucket_name)           # Bucket em que será salvo a midia
    
    if mime_type == "image/jpeg":
        extension = "jpeg"
    elif mime_type == "image/png":
        extension = "png"
    elif mime_type == "image/webp":
        extension = "webp"

    file_name = f"{tel}_{int(time.time())}.{extension}"             # Nome do arquivo a ser salvo (concatena número do telefone + timestamp)
    blob = bucket.blob(file_name)               
    try:
        blob.upload_from_string(media, content_type=mime_type)    # Realiza o Upload do arquivo para o Cloud Storage      
        return file_name        
    except Exception as e:
        error_message = f"Erro ao tentar salvar IMAGEM no Bucket da Google Cloud Storage. Detalhes: {e}"
        insert_internal_error("store_image", error_message, tel)
        return False

# Salva mensagem em banco No-SQL para recuperação de histórico de conversa
def store_message(tel, role, message):
    try:
        doc_ref = db.collection(f"message_history_{tel}").document()
        doc_ref.set({
            "timestamp": int(time.time()),
            "role": role,
            "parts": [message]
        })    
    except Exception as e:
        print(f"Erro ao salvar mensagem no Firebase/FireStore. Detalhes: {e}")
        return False
    
# Salva id da Midia recebida 
def store_idMedia(id_media):
    doc_ref = db.collection("id_medias").document()
    doc_ref.set({
            "timestamp": int(time.time()),
            "id_media": id_media
        })    

# Verifica se mídia já foi recebida - para evitar duplicidades
def exist_idMedia(id_media):
    mensagens_ref = db.collection("id_medias").where("id_media", "==", id_media)
    mensagens = mensagens_ref.stream()
    response = False
    for mensagem in mensagens:
        response = True    
    return response

# Salva id do texto recebido
def store_idText(id_text):
    doc_ref = db.collection("id_text").document()
    doc_ref.set({
            "timestamp": int(time.time()),
            "id_text": id_text
        })   
    
# Verifica se texto já foi recebido - para evitar duplicidades
def exist_idText(id_text):
    mensagens_ref = db.collection("id_text").where("id_text", "==", id_text)
    mensagens = mensagens_ref.stream()
    response = False
    for mensagem in mensagens:
        response = True    
    return response

# Obtem histórico de mensagens do telefone, a partir de Banco No-SQL hospedado na Google Cloud FireStore/Firebase
def get_menssages(tel):
    mensagens_ref = db.collection(f"message_history_{tel}").order_by("timestamp")
    mensagens = mensagens_ref.stream()

    # Lista para armazenar as mensagens formatadas
    messages_array = []

    for mensagem in mensagens:
        message_dict = mensagem.to_dict()
        # Cria o formato desejado para cada mensagem
        formatted_message = {
            "role": message_dict["role"],
            "parts": message_dict["parts"]
        }
        messages_array.append(formatted_message)

    return messages_array

# Salvar notificações da WhatsApp Cloud API recebidas e não analisada 
def untreated_notification(doc):
    doc_ref = db.collection("notifications").document()
    doc_ref.set(doc)

# Verifica se contato existe
def exist_contact(tel):
    response = False
    docs = db.collection("contacts").where("Telefone", "==", tel).stream()
    for doc in docs:
        response = True
    return response    

# Salva contato
def store_contact(tel):    
    doc_ref = db.collection("contacts").document()
    doc_ref.set({
            "Telefone": tel,
            "Nome": "",
            "Nascimento": "",
            "Sexo": "",
            "Ocupacao": "",
            "Bairro": "",
            "Tipo_Contato": "Ativos",
            "timestamp": int(time.time())
        })

#  Atualiza Contato
def update_contact(tel, nome, bairro, nascimento, sexo, ocupacao):
    doc_ref = db.collection("contacts").where("Telefone", "==", tel).get()    
    new_doc = {'Nome': nome, 'Nascimento' : nascimento, 'Sexo': sexo, 'Ocupacao': ocupacao, 'Bairro': bairro}  
    if doc_ref:
        doc_ref[0].reference.update(new_doc)

#  Atualiza Contato com File_Name da ultima imagem recebida 
def update_contact_last_media(tel, file_name):
    doc_ref = db.collection("contacts").where("Telefone", "==", tel).get()    
    update_doc = {'last_media': file_name}  
    if doc_ref:
        doc_ref[0].reference.update(update_doc)

# Obtem file_name da ultima imagem recebida, referente ao novo chamado
def get_contact_last_media(tel):
    last_media = ""
    doc_ref = db.collection("contacts").where("Telefone", "==", tel).get()    
    if doc_ref:
        media_dict = doc_ref[0].to_dict()
        last_media = media_dict["last_media"]
    return last_media

# Verifica se existem instruções ou comandos enviados pela IA e faz a devida separação da mensagem
def response_treatment(message: str):
    treatad_message = message
    instruction = ""
    
    instruction_position = message.find("#SmartChat")    
    end_instruction_position = message.find("#End")  

    if instruction_position != -1:
        instruction = message[instruction_position:end_instruction_position]                

    return treatad_message, instruction

# Analisa instrução e dá o devido tratamento
# Para acréscimos de novas instruções devem ser implementados ajustes no Prompt e no código
def handle_instruction(instruction: str, tel):
    details = instruction.split("#")
    instruction_type = details[2]
   
    # Instrução referente a Registro de Chamados
    if instruction_type == "NewRequest":
        try:
            media = get_contact_last_media(tel)     # Obtem file_name da ultima imagem recebida, referente ao novo chamado
            colection = "requests"
            timestamp = int(time.time())
            nome = details[3]
            nascimento = details[4]
            sexo = details[5]
            ocupacao = details[6]
            bairro = details[7]
            descricao = details[8]
            setor = details[9]            
            data = {
                "id": f"{tel}_{timestamp}",
                "nome": nome,
                "nascimento": nascimento,
                "sexo": sexo,
                "ocupacao": ocupacao,
                "tel": tel,
                "bairro": bairro,
                "descricao": descricao,
                "setor": setor,
                "status": "EM ANÁLISE",
                "timestamp": timestamp,
                "media": media
            }                          
            insert_request(colection, data)      
            update_contact(tel, nome, bairro, nascimento, sexo, ocupacao)  
        except Exception as e:
            insert_internal_error("requests", e, tel)        
    else:
        insert_internal_error("requests", f"Instrução desconhecida - {instruction}", tel)
        
# Insere Chamado em Banco de Dados
def insert_request(colection, data):
    doc_ref = db.collection(colection).document()
    doc_ref.set(data)

# Insere eventuais registros de erro em banco NoSQL
def insert_internal_error(operation, error_message, tel):
    data ={
        "tel": tel,
        "operation": operation,
        "error": error_message,
        "timestamp": int(time.time()),
    }
    doc_ref = db.collection("internal_error").document()
    doc_ref.set(data)

#  Atualiza Status de Campanhas Promocionais
def update_campaign_status(event, message_template_id, reason):
    if reason == None: reason = ""
    doc_ref = db.collection("campaigns").where("ID_Modelo", "==", f"{message_template_id}").get()    
    new_doc = {'Status': event, "Obs": reason}    
    if doc_ref:        
        doc_ref[0].reference.update(new_doc)

def get_campaign_name(message_template_id):
    response = ""
    docs = db.collection("campaigns").where("ID_Modelo", "==", f"{message_template_id}").get()
    if docs:
        media_dict = docs[0].to_dict()
        response = media_dict["Nome_Campanha"].upper()
    return response 

# Salva Alertas relativos a Campanhas Promocionais
def store_campaign_alert(alert):
    doc_ref = db.collection("alerts").document()
    doc_ref.set({
            "Descricao": alert,
            "timestamp": int(time.time())
        })

# Salva json de requisição recebida. Para efeito de depuração
def store_json(data):
    doc_ref = db.collection("log_messages").document()
    doc_ref.set(data)

# Inativar Cadastro
def contact_update_status(tel, new_status):
    doc_ref = db.collection("contacts").where("Telefone", "==", tel).get()    
    new_doc = {'Tipo_Contato': new_status}  
    if doc_ref:
        doc_ref[0].reference.update(new_doc)

# Atualiza Score de Campanhas Promocionais
def update_campaign_score(message_template_id, previus, new):
    doc_ref = db.collection("campaigns").where("ID_Modelo", "==", f"{message_template_id}").get()    
    new_doc = {'Score_qualidade_atual': new, "Score_qualidade_anterior": previus}    
    if doc_ref:        
        doc_ref[0].reference.update(new_doc)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))