
# Imagens de Satélite e Machine Learning para Previsão de Indicadores Socioeconômicos no Brasil

## Introdução

Repositório com o código para replicação dos resultados obtidos no projeto de mestrado "Imagens de Satélite e Machine Learning para Previsão de Indicadores Socioeconômicos no Brasil", que esta sendo realizado no Instituto de Ciências Matemáticas e de Computação - Universidade de São Paulo (ICMC - USP) pelo aluno João Pedro da Silva (jp.silva@usp.br).

Este trabalho tem como objetivo combinar imagens diurnas e noturnas de satélite, e aplicar a técnica de Transfer Learning para previsão de indicadores socioeconômicos ainda não cobertos pela literatura.

O projeto se baseia no artigo [Predicting socioeconomic indicators using transfer learning on imagery data: an application in Brazil](https://link.springer.com/article/10.1007/s10708-022-10618-3) e esta em andamento.

Estão sendo utilizados modelos de rede VGG16 e Resnet50, além de regressão Lasso e Ridge, aplicados a um conjunto imagens de satélite de 140 cidades de todo o Brasil para prever indicadores como número populacional e renda média das cidades.

Os passos a seguir mostram como repoduzir os resultados preliminares.
  
## Download data

+ Donwload da imagem noturna (NTL): https://eogdata.mines.edu/nighttime_light/annual/v21/2021/

Os demais arquivos de entrada já estão presentes no repositório, mas podem ser acessados através dos seguintes links:

+ Indicadores:
	+ População (Censo 2022): https://www.ibge.gov.br/estatisticas/sociais/populacao/22827-censo-demografico-2022.html?edicao=35938&t=resultados
	+ Projeção de Renda média: https://www.cps.fgv.br/cps/bd/docs/ranking/TOP_Municipio2020.htm
+ Shapefiles do Brasil e Bahia: https://portaldemapas.ibge.gov.br/ (clicando em "Organização do Território" -> "Malhas territoriais" -> "Malha de municípios")
+ Coordenadas de cada município do Brazil (centros de cidades): ftp://geoftp.ibge.gov.br/estrutura_territorial/localidades/Shapefile_SHP/BR_Localidades_2010_v1.shp

## Scripts de Pré processamento de dados

+ Na pasta "scripts", executar na seguinte ordem: 
	+ todos que começam com "prepare_data_*"
	+ "get_nighttime_lights": Extrair caracteristicas da imagem de luz noturna (radiancia e geolocalização)
	+ "associate_cities_nightlights": associar os pontos de luz noturna com cada município
	+ "get_all_cities_coordinates": gerar um arquivo com a geolocalização de todos os centros de cidades da Bahia.
	+ "nightlights_distances_per_city": calcular a distancia euclidiana entre os pontos de luz noturna e o centro da cidade.
	+ "nearest_nightlights_per_city": filtrar apenas os 100 pontos mais próximos do centro da cidade.

## Obter Google Maps Static images

Após executar os scripts anteriores é necessário fazer o download das imagens diurnas de satélite no  Google Maps Static images:

+ Criar uma conta (ou utilizar um já existente) no Google Cloud
+ Adicionar uma chave de API para o "Maps API Key"
+ Em "scripts", executar o arquivo "download_google_images.py" substituindo a Key do Google Maps obtido no console.

## Extrair features

Para extrair features das imagens diurnas de satélite, utilizar os scripts da pasta "extract-features", onde existem alguns modelos disponíveis para extração.

## Regressão

Para aplicar uma regressão sobre as features extraídas anteriormente, com a intenção de previsão dos indicadores socioeconômicos, utilize os scripts da pasta "regression".
