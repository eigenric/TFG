import requests
from bs4 import BeautifulSoup
import json
import os
import time
import base64
from pydub import AudioSegment
import re
import traceback

# --- Configuración ---
# !!! IMPORTANTE: Reemplaza con la ruta a tu archivo JSON de credenciales !!!
GOOGLE_APPLICATION_CREDENTIALS_FILE = "credenciales.json"
# !!! IMPORTANTE: Reemplaza con el ID de tu proyecto de Google Cloud !!!
GCLOUD_PROJECT_ID = "heroic-bird-459121-h1" # Ej: "heroic-bird-459121-h1"

INPUT_MARKDOWN_FILE = "guion_silencio.md"
OUTPUT_DIR = "guion_silencio"
MAX_TEXT_CHUNK_SIZE = 4800 

# --- NUEVA CONFIGURACIÓN DE SILENCIO ---
# Duración del silencio en milisegundos a insertar entre diapositivas (---).
# 2000ms = 2 segundos. Cambia este valor según necesites.
SILENCE_BETWEEN_SLIDES_MS = 1500

# Marcador interno para las pausas. No es necesario cambiarlo.
SLIDE_BREAK_MARKER = "##SLIDE_BREAK_PLACEHOLDER##"

VOICE_CONFIG = {
    "languageCode": "es-ES",
    "name": "es-ES-Chirp3-HD-Fenrir",
}
AUDIO_CONFIG = {
    "audioEncoding": "MP3",
    "sampleRateHertz": 24000
}

REQUEST_TIMEOUT = 90
MAX_SYNTHESIS_RETRIES = 3
RETRY_SLEEP_BASE = 5
INTER_CHUNK_SLEEP = 1.0

DEBUG_MODE = True # Cambia a False para menos verbosidad

# --- Autenticación con Cuenta de Servicio ---
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest

_service_account_token = None
_service_account_token_expiry = 0

def get_access_token_from_service_account():
    global _service_account_token, _service_account_token_expiry
    current_time = time.time()

    if not _service_account_token or current_time >= (_service_account_token_expiry - 120):
        if DEBUG_MODE: print("DEBUG: Obteniendo/Refrescando token de acceso desde cuenta de servicio...")
        try:
            scopes = ['https://www.googleapis.com/auth/cloud-platform']
            credentials = service_account.Credentials.from_service_account_file(
                GOOGLE_APPLICATION_CREDENTIALS_FILE, scopes=scopes)
            
            if not credentials.valid:
                 auth_req = GoogleAuthRequest()
                 credentials.refresh(auth_req)

            _service_account_token = credentials.token
            _service_account_token_expiry = credentials.expiry.timestamp() if credentials.expiry else current_time + 3500 
            if DEBUG_MODE: print(f"DEBUG: Token de cuenta de servicio obtenido/refrescado. Válido hasta: {time.ctime(_service_account_token_expiry)}")
        except Exception as e:
            print(f"Error al obtener/refrescar token de cuenta de servicio: {e}")
            traceback.print_exc()
            _service_account_token = None
            raise
    return _service_account_token

# --- Funciones Auxiliares ---

def sanitize_filename(name):
    name = re.sub(r'[^\w\s.-]', '', name)
    name = re.sub(r'[-\s]+', '_', name).strip('_')
    return name

def clean_chapter_text(raw_text):
    """
    Limpia el texto del capítulo. Esta función ahora se ejecuta sobre texto
    que ya tiene el marcador de diapositiva (SLIDE_BREAK_MARKER).
    """
    if not raw_text:
        return ""

    text_no_bold_markers = re.sub(r'\*\*(.*?)\*\*', r'\1', raw_text)
    text = re.sub(r'[ \t]+', ' ', text_no_bold_markers)
    text = text.strip()
    
    placeholder = "##NEWLINE_PLACEHOLDER##"
    text = re.sub(r'([.!?])\n', rf'\1{placeholder}', text)
    text = re.sub(r'\n(\s*\n)+', placeholder, text)
    text = text.replace('\n', ' ')
    text = text.replace(placeholder, '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

def parse_markdown_chapters(filepath):
    chapters = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: El archivo {filepath} no fue encontrado.")
        return chapters

    # MODIFICADO: Reemplaza '---' en una línea por el marcador de pausa.
    # Se usa regex para ser más robusto (ignora espacios en blanco alrededor).
    content = re.sub(r'^\s*---\s*$', SLIDE_BREAK_MARKER, content, flags=re.MULTILINE)
    if DEBUG_MODE:
        if SLIDE_BREAK_MARKER in content:
            print(f"DEBUG: Se encontraron y reemplazaron separadores '---' por '{SLIDE_BREAK_MARKER}'.")
        else:
            print("DEBUG: No se encontraron separadores '---' en el guion.")


    chapter_pattern = re.compile(
        r'^\+\+\+\s*title\s*=\s*"(.*?)"\s*weight\s*=\s*\d+\s*\+\+\+$(.*?)(?=(?:^\+\+\+\s*title|\Z))',
        re.MULTILINE | re.DOTALL
    )
    
    matches = list(chapter_pattern.finditer(content))
    if not matches:
        if DEBUG_MODE: print("DEBUG: No se encontraron capítulos con el patrón regex principal. Intentando método alternativo...")
        parts = re.split(r'(\n\s*\+\+\+\s*title\s*=\s*".*?"\s*weight\s*=\s*\d+\s*\+\+\+\s*\n)', content)
        current_title = "Contenido_Inicial_Sin_Titulo_Asignado"
        title_regex_alt = re.compile(r'title\s*=\s*"(.*?)"')

        if parts[0].strip():
            soup = BeautifulSoup(parts[0], 'html.parser')
            raw_chapter_text = soup.get_text(separator='\n', strip=True)
            cleaned_text = clean_chapter_text(raw_chapter_text)
            if cleaned_text:
                 chapters.append((current_title, cleaned_text))
                 if DEBUG_MODE: print(f"DEBUG: Parseado capítulo inicial. Original {len(raw_chapter_text)} chars, Limpio {len(cleaned_text)} chars.")
        
        for i in range(1, len(parts), 2):
            header_part = parts[i]
            content_part = parts[i+1] if (i+1) < len(parts) else ""
            title_match_in_part = title_regex_alt.search(header_part)
            if title_match_in_part:
                current_title = title_match_in_part.group(1).strip()
            else:
                current_title = f"Capitulo_Sin_Titulo_{len(chapters)+1}"
            
            soup = BeautifulSoup(content_part, 'html.parser')
            raw_chapter_text = soup.get_text(separator='\n', strip=True)
            cleaned_text = clean_chapter_text(raw_chapter_text)
            chapters.append((current_title, cleaned_text))
            if DEBUG_MODE: print(f"DEBUG: Parseado capítulo '{current_title}'. Original {len(raw_chapter_text)} chars, Limpio {len(cleaned_text)} chars (método alternativo).")
            
        if chapters and chapters[0][0] == "Contenido_Inicial_Sin_Titulo_Asignado" and not chapters[0][1].strip():
            chapters.pop(0)
    else:
        for match_idx, match in enumerate(matches):
            title = match.group(1).strip()
            chapter_text_html = match.group(2).strip()
            soup = BeautifulSoup(chapter_text_html, 'html.parser')
            raw_chapter_text = soup.get_text(separator='\n', strip=True)
            
            if DEBUG_MODE and match_idx < 1:
                print(f"DEBUG parse_markdown_chapters: Texto crudo (primeros 300 chars) para '{title}':\n'{raw_chapter_text[:300]}'")

            cleaned_text = clean_chapter_text(raw_chapter_text)

            if DEBUG_MODE and match_idx < 1:
                print(f"DEBUG parse_markdown_chapters: Texto limpio (primeros 300 chars) para '{title}':\n'{cleaned_text[:300]}'")
            
            chapters.append((title, cleaned_text))
            if DEBUG_MODE: print(f"DEBUG: Parseado capítulo '{title}'. Original {len(raw_chapter_text)} chars, Limpio {len(cleaned_text)} chars (método principal).")
            
    return chapters

def split_text(text, max_length):
    # Esta función ahora no necesita conocer el marcador de diapositiva.
    # Se aplicará al contenido de cada diapositiva individualmente.
    if DEBUG_MODE: print(f"DEBUG split_text: Iniciando división de texto con longitud {len(text)} y max_length {max_length}")
    paragraphs_from_input = text.split('\n') 
    chunks = []
    current_chunk_content = ""

    for p_idx, p_text in enumerate(paragraphs_from_input):
        p_processed = p_text.strip() 

        if DEBUG_MODE and p_idx < 5 : print(f"DEBUG split_text: Procesando párrafo/línea {p_idx}: '{p_processed[:60]}...' (longitud: {len(p_processed)})")

        if not p_processed:
            if current_chunk_content.strip():
                current_chunk_content += "\n" 
            continue

        separator = "\n" if current_chunk_content and not current_chunk_content.endswith('\n') else ""
        len_if_added = len(current_chunk_content) + len(separator) + len(p_processed)

        if len_if_added <= max_length:
            current_chunk_content += separator + p_processed
        else:
            if current_chunk_content.strip():
                chunks.append(current_chunk_content.strip())
            
            current_chunk_content = p_processed 
            
            while len(current_chunk_content) > max_length:
                slice_to_examine = current_chunk_content[:max_length]
                split_point = -1
                sentence_enders = ['. ', '! ', '? ']
                other_breaks = ['\n']

                for sep in sentence_enders:
                    idx = slice_to_examine.rfind(sep)
                    if idx != -1:
                        split_point = idx + len(sep)
                        break
                
                if split_point == -1:
                    for sep in other_breaks:
                        idx = slice_to_examine.rfind(sep)
                        if idx != -1:
                            split_point = idx + len(sep)
                            break
                
                if split_point == -1: 
                    idx = slice_to_examine.rfind(' ')
                    if idx != -1:
                        split_point = idx + 1
                    else:
                        split_point = max_length
                
                if split_point == 0 and max_length > 0: 
                     split_point = max_length

                chunk_to_add = current_chunk_content[:split_point].strip()
                if chunk_to_add: 
                    chunks.append(chunk_to_add)
                
                current_chunk_content = current_chunk_content[split_point:].lstrip()
    
    if current_chunk_content.strip():
        chunks.append(current_chunk_content.strip())
    
    if DEBUG_MODE: print(f"DEBUG split_text: División completada. Total de chunks generados para este bloque: {len(chunks)}")
    return chunks

def synthesize_text_chunk(text_chunk, project_id, access_token_provider_func, chunk_idx_for_log="N/A"):
    tts_url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    
    for attempt in range(MAX_SYNTHESIS_RETRIES):
        if DEBUG_MODE: print(f"DEBUG synthesize: Chunk {chunk_idx_for_log}, Intento {attempt+1}/{MAX_SYNTHESIS_RETRIES}")
        try:
            current_token = access_token_provider_func() 
            if not current_token:
                print(f"Intento {attempt+1} (Chunk {chunk_idx_for_log}): No se pudo obtener token. Saltando.")
                return None

            headers = {
                "Content-Type": "application/json",
                "X-Goog-User-Project": project_id,
                "Authorization": f"Bearer {current_token}"
            }
            input_data = {"text": text_chunk}
            data = {
                "input": input_data,
                "voice": VOICE_CONFIG,
                "audioConfig": AUDIO_CONFIG
            }

            response = requests.post(tts_url, headers=headers, data=json.dumps(data), timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            response_json = response.json()
            if "audioContent" in response_json:
                if DEBUG_MODE: print(f"DEBUG synthesize: Chunk {chunk_idx_for_log}, Síntesis exitosa en intento {attempt+1}.")
                return base64.b64decode(response_json["audioContent"])
            else:
                print(f"Intento {attempt+1} (Chunk {chunk_idx_for_log}): Respuesta API sin 'audioContent'. Detalles: {response_json}")
                return None

        except requests.exceptions.HTTPError as http_err:
            error_message = f"Intento {attempt+1} (Chunk {chunk_idx_for_log}): Error HTTP "
            response_text_content = "No se pudo obtener el cuerpo de la respuesta."
            
            if http_err.response is not None:
                error_message += f"{http_err.response.status_code} "
                try:
                    response_text_content = json.dumps(http_err.response.json(), indent=2)
                except json.JSONDecodeError:
                    response_text_content = http_err.response.text 
            else:
                 error_message += "N/A "
            
            error_message += f"en API TTS: {http_err}"
            print(error_message)
            print(f"CUERPO RESPUESTA ERROR (Chunk {chunk_idx_for_log}):\n{response_text_content}\n")
            
            should_retry = False
            if http_err.response is not None and http_err.response.status_code in [401, 403, 429, 500, 502, 503, 504]:
                should_retry = True
            
            if "INVALID_ARGUMENT" in response_text_content and ("Input text not set" in response_text_content or "exceeds limit" in response_text_content):
                print(f"Error específico (Chunk {chunk_idx_for_log}): Argumento inválido. El chunk puede estar vacío o exceder el límite de 5000 bytes.")
                print(f"Texto del chunk (primeros/últimos 50 chars): '{text_chunk[:50]}...{text_chunk[-50:]}'")
                return None 

            if should_retry and attempt < MAX_SYNTHESIS_RETRIES - 1:
                wait_time = RETRY_SLEEP_BASE * (2 ** attempt) 
                print(f"Reintentando Chunk {chunk_idx_for_log} en {wait_time} segundos...")
                time.sleep(wait_time)
            elif not should_retry:
                 print(f"Error no recuperable para Chunk {chunk_idx_for_log}, no se reintentará.")
                 return None
            else:
                 print(f"Se agotaron los reintentos para Chunk {chunk_idx_for_log}.")
                 return None
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
            print(f"Intento {attempt+1} (Chunk {chunk_idx_for_log}): Error de conexión/timeout API TTS: {conn_err}")
            if attempt < MAX_SYNTHESIS_RETRIES - 1:
                wait_time = RETRY_SLEEP_BASE * (2 ** attempt)
                print(f"Reintentando Chunk {chunk_idx_for_log} en {wait_time} segundos...")
                time.sleep(wait_time)
            else:
                return None

        except Exception as e:
            print(f"Intento {attempt+1} (Chunk {chunk_idx_for_log}): Error inesperado durante síntesis: {e}")
            print(f"Texto del chunk problemático (primeros 100 chars): {text_chunk[:100]}")
            traceback.print_exc()
            return None 

    print(f"Todos los {MAX_SYNTHESIS_RETRIES} intentos de síntesis fallaron para el fragmento {chunk_idx_for_log}.")
    return None

# --- Flujo Principal ---
if __name__ == "__main__":
    print("--- Iniciando Generador de Audiolibro ---")
    if DEBUG_MODE: print("MODO DEBUG ACTIVADO: Se mostrará información detallada.")

    if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS_FILE):
        print(f"Error Crítico: Archivo de credenciales JSON no encontrado: {GOOGLE_APPLICATION_CREDENTIALS_FILE}")
        exit(1)
    if GCLOUD_PROJECT_ID == "tu-gcloud-project-id" or not GCLOUD_PROJECT_ID :
        print(f"Error Crítico: Debes configurar tu GCLOUD_PROJECT_ID en el script.")
        exit(1)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directorio de salida creado: {os.path.abspath(OUTPUT_DIR)}")

    try:
        print("Verificando autenticación inicial con cuenta de servicio...")
        get_access_token_from_service_account() 
        print(f"Autenticación con cuenta de servicio OK para proyecto: {GCLOUD_PROJECT_ID}")
    except Exception as e:
        print(f"Error Crítico en configuración inicial con credenciales: {e}")
        exit(1)

    print(f"Parseando archivo Markdown: {INPUT_MARKDOWN_FILE}")
    parsed_chapters = parse_markdown_chapters(INPUT_MARKDOWN_FILE)

    if not parsed_chapters:
        print(f"No se pudieron parsear capítulos de {INPUT_MARKDOWN_FILE}. Saliendo.")
        exit(1)

    total_chapters = len(parsed_chapters)
    print(f"Se encontraron {total_chapters} capítulos. (Después de la limpieza de texto)")
    all_chapter_final_audio_files = []

    for i, (chapter_title, chapter_content) in enumerate(parsed_chapters):
        chapter_num_log = f"({i+1}/{total_chapters})"
        original_chapter_title_for_log = chapter_title if chapter_title else f"Capitulo_Indice_{i+1}"
        sanitized_chapter_title = sanitize_filename(original_chapter_title_for_log)
        chapter_output_dir = os.path.join(OUTPUT_DIR, sanitized_chapter_title)
        
        audio_ext = AUDIO_CONFIG["audioEncoding"].lower()
        if audio_ext == "linear16": audio_ext = "wav"
        elif audio_ext == "ogg_opus": audio_ext = "ogg"
        
        final_chapter_audio_path = os.path.join(OUTPUT_DIR, f"{i:02d}_{sanitized_chapter_title}.{audio_ext}")

        print(f"\n--- Procesando Capítulo {chapter_num_log}: {original_chapter_title_for_log} ---")
        if DEBUG_MODE: 
            print(f"DEBUG: Longitud del contenido del capítulo (después de clean_chapter_text): {len(chapter_content)} caracteres.")
            print(f"DEBUG: Ruta final esperada para el capítulo: {final_chapter_audio_path}")

        if os.path.exists(final_chapter_audio_path) and os.path.getsize(final_chapter_audio_path) > 100: 
            print(f"  INFO: Archivo combinado del capítulo '{original_chapter_title_for_log}' ya existe. Omitiendo: {final_chapter_audio_path}")
            all_chapter_final_audio_files.append(final_chapter_audio_path)
            continue

        if not os.path.exists(chapter_output_dir):
            os.makedirs(chapter_output_dir)
            if DEBUG_MODE: print(f"DEBUG: Directorio de chunks para capítulo creado: {chapter_output_dir}")

        if not chapter_content.strip():
            print(f"  ADVERTENCIA: Contenido vacío para capítulo '{original_chapter_title_for_log}'. Omitiendo.")
            continue

        # --- LÓGICA MODIFICADA PARA DIAPOSITIVAS Y SILENCIOS ---
        slides = chapter_content.split(SLIDE_BREAK_MARKER)
        print(f"  Capítulo dividido en {len(slides)} diapositivas (basado en '---').")
        
        combined_chapter_audio = AudioSegment.empty()
        all_chunk_files_for_chapter = []
        global_chunk_counter = 0

        for slide_idx, slide_content in enumerate(slides):
            slide_num_log = f"(Diapositiva {slide_idx + 1}/{len(slides)})"
            
            if not slide_content.strip():
                if DEBUG_MODE: print(f"DEBUG: {slide_num_log} está vacía, omitiendo.")
                continue

            print(f"\n  Procesando {slide_num_log} para '{original_chapter_title_for_log}'...")
            
            print(f"    Dividiendo texto de la diapositiva en fragmentos (max_size={MAX_TEXT_CHUNK_SIZE} chars)...")
            text_chunks = split_text(slide_content, MAX_TEXT_CHUNK_SIZE)
            total_chunks_for_slide = len(text_chunks)
            print(f"    Texto de la diapositiva dividido en {total_chunks_for_slide} fragmentos.")
            
            slide_audio = AudioSegment.empty()
            
            for j, chunk_text in enumerate(text_chunks):
                chunk_num_log = f"({j+1}/{total_chunks_for_slide})"
                print(f"    Procesando fragmento {chunk_num_log} de la diapositiva (longitud: {len(chunk_text)} chars)...")
                
                # Preview del texto
                preview_text = chunk_text.replace('\n', ' <NL> ')
                max_preview_len = 70
                preview_display = f"'{preview_text[:max_preview_len//2]}...{preview_text[-(max_preview_len//2):]}'" if len(preview_text) > max_preview_len else f"'{preview_text}'"
                print(f"      Texto: {preview_display}")

                if not chunk_text.strip():
                    print(f"      ADVERTENCIA: Fragmento {chunk_num_log} está vacío. Omitiendo.")
                    continue
                
                # Usar un contador global para que los nombres de archivo sean únicos en todo el capítulo
                chunk_file_name = f"chunk_{global_chunk_counter:03d}.{audio_ext}"
                chunk_file_path = os.path.join(chapter_output_dir, chunk_file_name)
                global_chunk_counter += 1
                all_chunk_files_for_chapter.append(chunk_file_path)

                if os.path.exists(chunk_file_path) and os.path.getsize(chunk_file_path) > 100: 
                    print(f"      INFO: Fragmento '{chunk_file_name}' ya existe. Omitiendo síntesis.")
                    audio_data = open(chunk_file_path, "rb").read()
                else:
                    print(f"      Sintetizando audio...")
                    audio_data = synthesize_text_chunk(chunk_text, GCLOUD_PROJECT_ID, get_access_token_from_service_account, chunk_idx_for_log=f"{original_chapter_title_for_log} - Slide {slide_idx+1} - Chunk {j+1}")
                
                if audio_data:
                    if not (os.path.exists(chunk_file_path) and os.path.getsize(chunk_file_path) > 100):
                        with open(chunk_file_path, "wb") as f:
                            f.write(audio_data)
                        print(f"      ÉXITO: Fragmento guardado en: {chunk_file_path}")
                    try:
                        segment = AudioSegment.from_file(chunk_file_path, format=audio_ext)
                        slide_audio += segment
                    except Exception as e:
                        print(f"    ERROR al cargar/combinar {chunk_file_path}: {e}. Omitiendo este fragmento.")
                else:
                    print(f"      FALLO: No se pudo sintetizar el fragmento.")
                    # Guardar el texto fallido
                    failed_chunk_text_file = os.path.join(chapter_output_dir, f"failed_chunk_{global_chunk_counter-1:03d}.txt")
                    try:
                        with open(failed_chunk_text_file, "w", encoding="utf-8") as f_err: f_err.write(chunk_text)
                    except Exception as e_write: print(f"ERROR al guardar texto fallido: {e_write}")
                    continue 
                
                if DEBUG_MODE and j < total_chunks_for_slide - 1: print(f"DEBUG: Pausa de {INTER_CHUNK_SLEEP}s antes del siguiente fragmento.")
                time.sleep(INTER_CHUNK_SLEEP)
            
            # Añadir el audio completo de la diapositiva al audio del capítulo
            if len(slide_audio) > 0:
                combined_chapter_audio += slide_audio
                if DEBUG_MODE: print(f"DEBUG: Audio de {slide_num_log} ({len(slide_audio)/1000:.2f}s) añadido al capítulo. Duración total ahora: {len(combined_chapter_audio)/1000:.2f}s")
            
            # Añadir silencio si no es la última diapositiva
            if slide_idx < len(slides) - 1:
                if SILENCE_BETWEEN_SLIDES_MS > 0:
                    silence = AudioSegment.silent(duration=SILENCE_BETWEEN_SLIDES_MS)
                    combined_chapter_audio += silence
                    print(f"  -> SILENCIO de {SILENCE_BETWEEN_SLIDES_MS}ms añadido después de la diapositiva {slide_idx + 1}.")
                    if DEBUG_MODE: print(f"DEBUG: Duración total del capítulo después del silencio: {len(combined_chapter_audio)/1000:.2f}s")

        # --- FIN DE LÓGICA MODIFICADA ---

        if len(combined_chapter_audio) > 0:
            print(f"\n  Exportando audio combinado final para '{original_chapter_title_for_log}'...")
            try:
                combined_chapter_audio.export(final_chapter_audio_path, format=audio_ext)
                print(f"  ÉXITO: Capítulo '{original_chapter_title_for_log}' guardado en: {final_chapter_audio_path}")
                all_chapter_final_audio_files.append(final_chapter_audio_path)
            except Exception as e:
                print(f"    ERROR al exportar audio combinado final para '{original_chapter_title_for_log}': {e}")
                traceback.print_exc()
        else:
            print(f"  INFO: No se generó audio para el capítulo '{original_chapter_title_for_log}'.")

    print("\n--- Proceso Completado ---")
    if all_chapter_final_audio_files:
        print("Archivos de audio de capítulos combinados generados:")
        for f_path in sorted(all_chapter_final_audio_files):
            print(f"  - {os.path.abspath(f_path)}")
    else:
        print("No se generaron archivos de audio de capítulos combinados.")
    print(f"Todos los chunks individuales están en subdirectorios dentro de: {os.path.abspath(OUTPUT_DIR)}")
    if DEBUG_MODE: print("MODO DEBUG ESTUVO ACTIVADO.")
    print("--- Fin del Generador de Audiolibro ---")