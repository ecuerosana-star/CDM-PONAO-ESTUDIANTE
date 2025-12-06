

from fastapi import FastAPI
import uvicorn 

app = FastAPI()


from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="API REST Cliente-Servidor",
    description="API base desarrollada con FastAPI, Docker y pruebas automáticas",
    version="1.0.0"
 )
class Usuario(BaseModel):
    id: int
    nombre: str
    correo: str

usuarios = [
    {"id": 1, "nombre": "Lisa", "correo": "lisa@example.com"},
    {"id": 2, "nombre": "Juan", "correo": "juan@example.com"},
]

@app.get("/")
def inicio():
    print("Servidor FastAPI funcionando correctamente ")
    return {"mensaje": "Bienvenido a la API REST Cliente-Servidor"}

@app.get("/usuarios")
def listar_usuarios():
    return {"usuarios": usuarios}

@app.post("/usuarios")
def crear_usuario(usuario: Usuario):
    usuarios.append(usuario.dict())
    print(f"Usuario agregado: {usuario.nombre}")
    return {"mensaje": "Usuario agregado correctamente", "usuario": usuario }
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
        

     