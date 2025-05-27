



document.body.innerHTML = "<h1>minimal signature UI</h1>"

async function load() {

  let doclist :string[] = []
  await fetch("http://localhost:5000/api/list_documents").then(async (response) => {
    
    await response.json().then((data) => {
      console.log(data);

      doclist = data;
    })
  });

  console.log(doclist);
  

  let doc_picker = document.createElement("select");
  doc_picker.id = "doc_picker";

  doclist.forEach((doc) => {
    let option = document.createElement("option");
    option.value = doc;
    option.textContent = doc;
    doc_picker.appendChild(option);

  });

  doc_picker.addEventListener("change", (event) => {
    let selectedDoc = (event.target as HTMLSelectElement).value;
    console.log("Selected document:", selectedDoc);
  });

  
  
  document.body.appendChild(doc_picker);

}

load()


export {}