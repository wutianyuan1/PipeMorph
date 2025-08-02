#include <hiredis/hiredis.h>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <sstream>
#include <stdexcept>
#include <cassert>


class Client {
private:
    redisContext* context;

    std::string _get_from_redis(std::string key, int* status) {
        std::string cmd = "GET " + key;
        redisReply* reply = (redisReply*)redisCommand(context, cmd.c_str());
        if (reply == nullptr) {
            printf("[RedisClient] NullReply Error: %s\n", context->errstr);
            *status = -1;
            return "Error";
        }

        std::string ret;
        if (reply->type == REDIS_REPLY_STRING) {
            ret = reply->str;
        } else {
            printf("[RedisClient] Unexpected reply type: %d\n", reply->type);
            *status = -1;
            ret = "Error";
        }

        freeReplyObject(reply);
        return ret;
    }

public:
    Client(const std::string address, int port) {
        struct timeval timeout = { 1, 500000 }; // 1.5 seconds
        context = redisConnectWithTimeout(address.c_str(), port, timeout);
        if (context == nullptr || context->err) {
            if (context) {
                printf("[RedisClient] Connection error: %s", context->errstr);
                redisFree(context);
            } else {
                printf("[RedisClient] Connection error: can't allocate redis context\n");
            }
            exit(1);
        }
    }

    ~Client() {
        if (context != nullptr) {
            redisFree(context);
        }
    }

    std::vector<float> get_sleep_time() {
        int status = 0;
        std::vector<float> ret;
        std::string reply = _get_from_redis("sleep_time", &status);
        if (status == -1)
            return ret;
        std::stringstream ss(reply);
        std::string token;

        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                ret.emplace_back(std::stof(token));
            }
        }
        return ret;
    }

    std::vector<std::tuple<int, int>> get_slow_links() {
        int status = 0;
        std::vector<std::tuple<int, int>> ret;
        std::string reply = _get_from_redis("slow_links", &status);
        if (status == -1)
            return ret;
        std::stringstream ss(reply);
        std::string token;

        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                std::stringstream pairStream(token);
                std::string part;
                int a, b;
                if (std::getline(pairStream, part, '_')) {
                    a = std::stoi(part);
                }
                if (std::getline(pairStream, part, '_')) {
                    b = std::stoi(part);
                }
                ret.emplace_back(a, b);
            }
        }
        return ret;
    }

    void get_if_nic_crash() {
        int status = 0;
        std::string reply = _get_from_redis("if_nic_crash", &status);
        if (status != -1) {
            std::string crashed = "yes";
            if (reply == crashed) {
                throw std::runtime_error("NIC Crash!");
            }
        }
    }
};
