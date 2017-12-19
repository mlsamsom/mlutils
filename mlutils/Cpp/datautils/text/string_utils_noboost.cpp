#include "string_utils_noboost.hpp"


namespace strutils
{
  template<typename T>
  void stringSplit(const std::string& inStr, char delim, T result)
  {
    std::stringstream ss(inStr);
    std::string item;
    while (std::getline(ss, item, delim))
      {
        *(result++) = item;
      }
  }

  std::vector<std::string> stringSplit(const std::string& inStr, char delim)
  {
    std::vector<std::string> tokens;
    stringSplit(inStr, delim, std::back_inserter(tokens));
    return tokens;
  }

}  // namespace strutils
