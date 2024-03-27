mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"youremail@domain\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

# Install JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64