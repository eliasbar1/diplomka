# README

Zde jsou shrnuty všechny soubory používané pro predikci využití nápovědy během hraní šifrovací hry společnosti Cryptomania. Pro tuto predikci je nejprve ohodnocena šifra pomocí modelu predikce obtížnosti, výstup z jeho předposlední vrstvy je následně použit v predikci společně s daty o týmu, šifře a hře.

> **POZOR** vzhledem k velikosti souborů s je složka **data** v ukázkové podobě a obsahuje jen jeden obrázek v každé ze složek, stejně tak **checkpoints**, které jsou ke stažení zde https://drive.google.com/file/d/1n_MpnyslmT0E-p9LqDx6zyixIo76ohE6/view?usp=sharing, https://drive.google.com/file/d/1qekyXbxr1hGDGx_P40Ea2MV3Stl8ybpe/view?usp=sharing 

Pro zobrazení vizualizace je třeba spustit webovou aplikaci pomocí příkazu `python ./dash_app.py` v kořenovém adresáři projektu. 

Její živý náhled je na sifry.okusovani.cz. Vzhledem k paměťové náročnosti výpočtů vyžaduje její použití trochu trpělivosti.



Hlavní soubor pro predikci využití nápovědy je **hint_prediction.ipynb**.

## Popis souborů

- složka **checkpoints**, která obsahuje váhy nejlepšího modelu predikce obtížnosti  a modelu predikce využití nápovědy
- složka **data**:
  - složka **legends** obsahující .json soubory s legendami jednotlivých her přiřazenými k šifrám
  - složka **original_trail_data** se záznamy průchodů týmů hrami Cryptomania
  - složka **output**, která obsahuje zdrojové obrázky šifer rozdělené na lehké a těžké a na trénovací a validační sadu, autorské šifry společnosti Cryptomania , které nejsou veřejně dostupné, byly odstaněny, protože jejich šíření je zakázáno, jsou zde tedy šifry převzaté z Facebooku firmy Cryptomania a z archivu šifrovací hry DNEM (https://www.chameleonbrno.org/dnem/kronika/kronika.html)
  - složka **saved** s podpůrnými soubory, které vznikly při předzpracování dat a urychlují konečnou predikci, zároveň jsou používány ve vizualizaci
  - **extracted_texts.json**: soubor s texty extrahovanými z obrázků šifer
  - **trail_dict_shortcuts.json**: soubor obsahující zkratky používané pro jednotlivé hry (*"BOB" : "3_bitva-o-brno"* ), tyto zkratky jsou používány např. v názvech obrázků
  - **trail_dict.json**: seznam her a jim přiřazených čísel, slouží pro párování hry a jejího názvu

- soubory pro zpracování dat z průchodu hrou:
  - **data_manager.py**: načítá data týmů a strukturu hry
  - **peak_computations.py**: soubor shromažďující funkce pro statistiky týmů, dobu trvání řešení šifry, které týmy šifru dokončily a podobně
  - **Trail.py**: třída jedné hry z pohledu hráčů, obsahuje jméno hry, výsledky týmů, a další související statistiky
  - **SetOfAssignments.py**: třída hry z pohledu šifer, obsahuje pořadí šifer, jednotlivé sady a návaznosti
- preprocessingové soubory:
  - **image_preprocessing.ipynb**: příprava obrázků (zmenšení, převod z .pdf do .png, přiřazení obtížnosti, extrakce textu), rozdělení na trénovací a validační sadu
  - **image_preprocess.py**: příprava obrázků k predikci, obsahuje funkce párující obrázek a jeho extrahovaný text pomocí jeho názvu,
  - **generate_image_embedding.ipynb**: vytvoření vektorové reprezentace obrázků, narozdíl od zbytku souborů je založený na knihovně *Torch* a ne na *TensorFlow*
  - **ImageEmbedding.py**: třída sloužící k vytvoření vektorové reprezentace obrázků
- **image_classification.ipynb**: učení a vyhodnocení modelu klasifikace obrázků (ten následně generuje vektorovou reprezentaci obrázků)
- **hint_data_preparation.ipynb**,   **ModelDataPreparator.py**: třída preparátoru a samotná příprava dat pro predikci nápovědy, převod na tensory a uložení, pro každý tým získá:
  - poměr dosud využitých nápověd a průměrný čas strávený řešením
  - pořadí šifry ve hře
  - určení hry
  - vektorovou reprezentaci obrazového zadání
  - legendu k šifře
  - predikovanou veličinu (vzal si tým nápovědu?)
- **MyModel.py**: obecná třída modelu, z něj dědí jednotlivé konkrétní modely,  ukládá váhy během učení a udržuje adresu jejich umístění
- soubory s definicí tříd modelů predikce obtížnosti šifry, liší se vstupy:
  - **ImageModelSingleLayer.py**: má na vstupu jen obrázky, model jednovrstvé konvoluční sítě sloužící jako baseline kvality predikce
  - **ImageTextModelFusion.py**: obrázek a z něj extrahovaný text
  - **TextEmbeddingModelFusion.py**: obrázek a jeho vektorová reprezentace
  - **ImageTextEmbeddingModelFusion.py**: obrázek, jeho vektorová reprezentace a extrahovaný  text, tato varianta modelu je použita ve vizualizaci
- soubory s definicí tříd modelů predikce využití nápovědy, liší se použitým jazykovým modelem:
  - **HintModel.py**: záleží na zvoleném modelu, buď multilingvní BERT, nebo multilingvní XLM-RoBERTa
  - **RobeczechModel.py**: RobeCzech specializovaný na texty v českém jazyce
- **hint_prediction.ipynb**: jupyter notebook se samotnou predikcí
- soubory pro webovou aplikaci:
  - **dash_app.py**: definice *Dash* rozhraní aplikace
  - **data_preparator.py**: funkce pro konstrukci vizualizací: generování dat ve formátu vhodném pro tvorbu grafů
  - **HintModelPreparator.py**: příprava dat pro vizualizace, převod na tensory a příprava podkladů pro predikční model
  - **index.html**, **index.js**: soubory zajišťující samotnou aplikaci, port, rozvržení apod.
- soubory pro generování nových obrázků
  - **helper_ds.py**: definice pomocných funkcí
  - **image_generating.ipynb**: generování nových obrázků

