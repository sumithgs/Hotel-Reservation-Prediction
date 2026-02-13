pipeline{
    agent any

    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins.......'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'githubtoken', url: 'https://github.com/sumithgs/Hotel-Reservation-Prediction.git']])
                }
            }
        }
    }
}