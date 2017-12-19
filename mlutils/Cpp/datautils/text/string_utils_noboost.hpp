/*
 * String utilities without boost dependencies incase you don't have root
 */
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>


namespace strutils
{

  /*
   * Splits a string on a delimiter like the python string method
   * @param inStr
   * @param delim
   * @param result
   */
  template<typename T>
  void stringSplit(const std::string& inStr, char delim, T result);

  /*
   * Splits a string on a delimiter like the python string method
   * and returns a vector of strings
   * @param inStr
   * @param delim
   */
  std::vector<std::string> stringSplit(const std::string& inStr, char delim);

} // namespace strutils
