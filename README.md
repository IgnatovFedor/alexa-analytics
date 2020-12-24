# alexa-analytics
Alexa Prize dialogue analysis tool


Чтобы заработало надо использовать у dp-agent ветку feat/dialog_list_api или смерджить эту ветку агента:
https://github.com/deepmipt/dp-agent/pull/99 (но могут упасть форматтеры по заявлению Паши)

Далее я поднимал агента из репа ассистента (https://github.com/deepmipt/assistant) из под ветки migration такой командой:

`python -m deeppavlov_agent.run -pl demo_stand_pipeline_conf.json -ch http_client --cors`

Когда у вас поднят агент или есть поднятый (можете исопльзовать его ip:port - http://0.0.0.0:4242).

Далее инсталяция-подъем админки:

- `git clone https://github.com/IgnatovFedor/alexa-analytics`
- `cd alexa-analytics`
- переключаетесь на ветку: feat/dp_agent_dumper
- pip install requirements.txt
- создаете базу постгре https://medium.com/coding-blocks/creating-user-database-and-adding-access-on-postgresql-8bfcd2f4a91e
- прописываете ее хосты-юзеров-базы-пароли в core/conf.json
- теперь можно дампать данные из агента и запускать админку:
- залить данные из базы агента в базу админки: `python -m core.run dpa_dumper -ac http://0.0.0.0:4242`
- запустить админку: `python -m core.run server -p 8001 -ac http://0.0.0.0:4242`
- Fin. (check http://0.0.0.0:8001/conversation/cf13b21c547e2655225ef41177044eb8 http://0.0.0.0:8001/admin/conversation/details/?id=cf13b21c547e2655225ef41177044eb8&url=%2Fadmin%2Fconversation%2F)

### kubernetes deployment

install postgresql helm chart
```
helm repo add bitnami https://charts.bitnami.com/bitnami
helm upgrade -i postgresql bitnami/postgresql -f ./helm/postgresql/values.yaml --set postgresqlPassword=<password>
```
create user for analytics database
```
kubernetes run -it --image=postgres postgres -- psql -h postgresql -U postgres
create user analytics with password <password>;
create database analytics with owner analytics;
grant all on database analytics to analytics;
```
build docker image
```
docker build -t <tag> .
docker push <tag>
```
install analytics helm chart
```
helm upgrade -i analytics ./helm/alexa-analytics --set image.repository=<tag> --set dbHost=<dbhost> --set dbUser=<dbuser> --set dbPassword=<dbpassword> --set dbName=<dbname> --set adminUser=<user> --set adminPassword=<password> --recreate-pods
```
