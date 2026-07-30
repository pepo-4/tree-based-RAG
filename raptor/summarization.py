import time
import json
import re
from typing import List, Dict, Tuple, Any, Optional
from transformers import AutoTokenizer
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from raptor.config import Config

SYSTEM_PROMPT = """
Sei un assistente specializzato nella sintesi tematica di testi per sistemi RAG gerarchici (RAPTOR).
Il tuo compito NON è riassumere ogni frammento in sequenza, ma identificare i temi, i pattern e i meccanismi comuni che attraversano l'intero gruppo di frammenti, producendo una sintesi concettuale unificata.
Devi essere accurato e non aggiungere mai informazioni non supportate dal contenuto fornito.
Devi attenerti STRETTAMENTE al formato richiesto, senza mai aggiungere introduzioni, commenti, spiegazioni sul tuo operato o frasi di chiusura.
"""

USER_PROMPT_TEMPLATE = """
I seguenti frammenti di testo appartengono a un unico gruppo tematico di bandi/documenti relativi a finanziamenti pubblici.

IL TUO COMPITO:
Genera un titolo generale e un riassunto che catturino il TEMA COMUNE e i PATTERN RICORRENTI condivisi dai frammenti, non un elenco descrittivo bando per bando.

COME STRUTTURARE IL RIASSUNTO:
1. Individua gli elementi trasversali che accomunano i frammenti (beneficiari, meccanismo di incentivo, finalità).
2. Organizza il discorso attorno a QUESTI elementi trasversali, non attorno all'elenco delle singole regioni/enti/bandi.
3. Cita regioni, enti o bandi specifici solo come esempi a supporto di un pattern, non come struttura portante di ogni frase.
4. Se più bandi condividono lo stesso meccanismo, raggruppali in un'unica frase invece di descriverli uno per uno.
5. Il Contenuto deve essere un UNICO blocco di prosa continua e scorrevole. NON usare grassetto, sottotitoli, intestazioni di sezione, o qualsiasi altra suddivisione visiva. NON usare elenchi puntati o numerati.

COME CHIUDERE IL TESTO (regola più importante):
Il "Contenuto" deve terminare su un dettaglio concreto e specifico — MAI su una frase che riepiloga, generalizza o commenta l'insieme di quanto appena scritto.

COSA EVITARE TASSATIVAMENTE:
- Elencare i frammenti nell'ordine in cui sono presentati.
- Frasi che si limitano a giustapporre informazioni senza collegarle concettualmente.
- NON citare mai l'etichetta numerica del frammento originale (es. "Frammento 3").

FORMATO RICHIESTO (rispondi ESCLUSIVAMENTE così):

Titolo: <titolo specifico e distintivo del cluster>

Contenuto: <riassunto unificato organizzato per pattern tematici, in prosa continua>

Testo completo dei frammenti da sintetizzare:
{context}
"""

def estrai_titolo_e_contenuto(testo: str) -> Tuple[str, str]:
    """Extracts title and content from LLM response text."""
    if not testo:
        return "", ""
        
    titolo = ""
    contenuto = testo.strip()
    
    match_titolo = re.search(r"^Titolo:\s*(.*)", testo, re.MULTILINE)
    if match_titolo:
        titolo = match_titolo.group(1).strip()
    
    # Estrazione del Contenuto
    match_contenuto = re.search(r"Contenuto:\s*(.*)", testo, re.DOTALL)
    if match_contenuto:
        contenuto = match_contenuto.group(1).strip()
    elif match_titolo:
        contenuto = testo.replace(match_titolo.group(0), "").strip()
        
    return titolo, contenuto

class GeminiClusterSummarizer:
    """LLM Summarizer using Google Gemini API with key rotation and progress saving."""
    
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model_name: str = "gemini-2.5-flash",
        output_progress_file: str = "riassunti_raptor_progress.json"
    ):
        self.api_keys = api_keys or Config.get_gemini_api_keys()
        self.model_name = model_name
        self.output_progress_file = output_progress_file
        self.clients = [genai.Client(api_key=k) for k in self.api_keys] if self.api_keys else []
        self.current_key_idx = 0
        self.tokenizer = AutoTokenizer.from_pretrained(Config.EMBEDDING_MODEL)

    def _get_client(self) -> Optional[genai.Client]:
        if not self.clients:
            return None
        return self.clients[self.current_key_idx]

    def _rotate_key(self):
        if self.clients:
            old_idx = self.current_key_idx
            self.current_key_idx = (self.current_key_idx + 1) % len(self.clients)
            print(f"🔄 Rotating Gemini API Key: {old_idx} -> {self.current_key_idx}")

    def generate_summary(self, context_text: str, retries: int = 3) -> Tuple[str, str]:
        """Generates summary using Gemini API with key rotation on rate limit / error."""
        if not self.clients:
            raise ValueError("No Gemini API keys configured. Set GEMINI_API_KEY in .env.")

        prompt = USER_PROMPT_TEMPLATE.format(context=context_text)

        for attempt in range(retries * len(self.clients)):
            client = self._get_client()
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                    )
                )
                raw_text = response.text or ""
                return estrai_titolo_e_contenuto(raw_text)
            except ClientError as e:
                print(f"⚠️ Gemini API ClientError (Key {self.current_key_idx}): {e}")
                self._rotate_key()
                time.sleep(2)
            except Exception as e:
                print(f"⚠️ Exception during Gemini generation: {e}")
                self._rotate_key()
                time.sleep(2)
                
        return "Sintesi Non Disponibile", "Impossibile generare la sintesi per questo cluster."
